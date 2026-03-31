#!/usr/bin/env python3
"""MLE-STAR Deep Agents Entry Point.

Usage:
    python -m mlestar.run --task <task_name> --data_dir <path>
    python -m mlestar.run --config config.yaml --task spaceship-titanic
"""

import os
import sys
import json
import yaml
import logging
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlestar.config import MleStarConfig
from mlestar.agent import create_mle_star_agent, configure_tools

start_ts = datetime.now()


def setup_logging(log_dir: Path, level: str = "INFO") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"mle_star_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def load_task_description(task_dir: str, task_name: str) -> str:
    possible_files = [
        os.path.join(task_dir, task_name, "description.txt"),
        os.path.join(task_dir, task_name, "research_problem.txt"),
        os.path.join(task_dir, task_name, "task.txt"),
        os.path.join(task_dir, task_name, "README.md"),
        os.path.join(task_dir, f"{task_name}.txt"),
    ]
    for fpath in possible_files:
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                return f.read()
    return f"Machine learning task: {task_name}"


def main():
    parser = argparse.ArgumentParser(
        description="MLE-STAR: Deep Agents Edition"
    )
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--data_dir", default=None, help="Path to data directory")
    parser.add_argument("--work_dir", default=None, help="Working directory")
    parser.add_argument("--log_dir", default=None, help="Log directory")
    parser.add_argument("--task_description", default=None, help="Task description string")
    parser.add_argument("--log_level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Load config
    config_path = args.config or Path(__file__).parent / "config.yaml"
    if os.path.exists(config_path):
        config = MleStarConfig.from_yaml(str(config_path))
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = MleStarConfig()

    # Override with CLI args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.work_dir:
        config.workspace_dir = args.work_dir
    if args.log_dir:
        config.log_dir = args.log_dir

    # Directories
    base_dir = Path(__file__).parent
    work_dir = Path(config.workspace_dir) / args.task
    log_dir = Path(config.log_dir) / args.task / datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = setup_logging(log_dir, args.log_level)
    logger = logging.getLogger("mle-star")

    logger.info("=" * 60)
    logger.info("MLE-STAR: Deep Agents Edition")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"LLM provider: {config.llm.provider}")
    logger.info(f"Params: M={config.mle_star.num_models_to_retrieve} "
                f"L={config.mle_star.num_parallel_solutions} "
                f"T={config.mle_star.max_ablation_iterations} "
                f"K={config.mle_star.max_refinement_iterations} "
                f"R={config.mle_star.max_ensemble_iterations}")
    logger.info(f"Sandbox: mode={config.sandbox.mode}, python={config.sandbox.python_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Config loaded from: {config_path}")


    # Task description
    if args.task_description:
        task_description = args.task_description
    else:
        task_dir = config.data_dir or ""
        task_description = load_task_description(task_dir, args.task)

    logger.info(f"Task description: {task_description[:200]}...")

    # Copy data to workspace
    if config.data_dir:
        input_dir = work_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        source_data = Path(config.data_dir) / args.task
        if source_data.exists():
            file_count = 0
            for item in source_data.iterdir():
                dest = input_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                    logger.info(f"Copied directory: {item.name}")
                else:
                    shutil.copy2(item, dest)
                    logger.info(f"Copied file: {item.name} ({item.stat().st_size / 1024:.1f} KB)")
                file_count += 1
            logger.info(f"Copied {file_count} items from {source_data} to {input_dir}")
        else:
            logger.warning(f"Source data directory not found: {source_data}")

    # Configure tools and create agent
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[run] Configuring tools...")
    configure_tools(config, str(work_dir))
    logger.info("[run] Creating MLE-STAR agent...")
    agent = create_mle_star_agent(config)
    start_ts = datetime.now()
    logger.info("[run] Agent created, starting invocation at %s", start_ts.strftime("%H:%M:%S"))

    # Build the user message with full context
    params = config.mle_star
    user_message = f"""# Task
{task_description}

# Workflow Parameters
- M (models to retrieve): {params.num_models_to_retrieve}
- L (parallel solutions): {params.num_parallel_solutions}
- T (ablation iterations): {params.max_ablation_iterations}
- K (refinement iterations): {params.max_refinement_iterations}
- R (ensemble iterations): {params.max_ensemble_iterations}

Execute the full 4-phase MLE-STAR workflow on this task."""

    try:
        logger.info("[run] Invoking agent with task message (%d chars)", len(user_message))
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"configurable": {"thread_id": f"mle-star-{args.task}"}},
        )

        # Extract results
        messages = result.get("messages", [])
        logger.info("[run] Agent completed — %d messages in conversation", len(messages))
        final_message = messages[-1].content if messages else "No result"

        results = {
            "success": True,
            "task": args.task,
            "final_message": final_message,
            "elapsed_time": (datetime.now() - start_ts).total_seconds(),
            "work_dir": str(work_dir),
        }

        elapsed = (datetime.now() - start_ts).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Elapsed: {elapsed:.1f}s")
        logger.info(f"Final output: {final_message[:500]}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        results = {"success": False, "error": str(e)}
        return 1

    # Save results
    results_file = log_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
