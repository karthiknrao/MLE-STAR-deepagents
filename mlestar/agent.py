"""MLE-STAR Deep Agent assembly.

Creates the root `create_deep_agent()` with 4 phase subagents,
custom tools, and provider-agnostic LLM initialization.
"""

import os
import logging
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent

from mlestar.config import MleStarConfig
from mlestar.prompts import load_prompts
import mlestar.tools.execute_code as ec_mod
from mlestar.tools.execute_code import (
    execute_code,
    execute_code_uncounted,
    save_code_artifact,
    get_sandbox_budget,
)
import mlestar.tools.web_search as ws_mod
from mlestar.tools.web_search import web_search
import mlestar.tools.file_utils as fu_mod
from mlestar.tools.file_utils import read_data_files, get_output_file

logger = logging.getLogger("mle-star.agent")


def _create_model(config: MleStarConfig) -> BaseChatModel:
    """Create the LLM model based on config provider."""
    llm = config.llm
    api_key = llm.get_api_key()

    logger.info("[model] Creating LLM — provider=%s, model=%s", llm.provider, llm.model)

    if llm.provider == "openrouter":
        logger.info("[model] Using OpenRouter (OpenAI-compatible) at https://openrouter.ai/api/v1")
        model = init_chat_model(
            model=llm.model,
            model_provider="openai",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            max_retries=6,
            timeout=120,
        )
    else:  # azure
        logger.info("[model] Using Azure OpenAI at %s", llm.azure_endpoint)
        model = init_chat_model(
            model=llm.model,
            model_provider="azure",
            azure_endpoint=llm.azure_endpoint,
            api_key=api_key,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            max_retries=6,
            timeout=120,
        )

    logger.info("[model] LLM created successfully")
    return model


def create_mle_star_agent(config: MleStarConfig):
    """Create and return the MLE-STAR Deep Agent."""
    logger.info("[agent] Loading prompts...")
    prompts = load_prompts()
    logger.info("[agent] Loaded %d prompts: %s", len(prompts), list(prompts.keys()))

    model = _create_model(config)

    # --- Subagent definitions ---
    logger.info("[agent] Defining 4 phase subagents...")

    initialization_subagent = {
        "name": "initialization",
        "description": (
            "Phase 1 of MLE-STAR: searches the web for M SOTA models, evaluates each by "
            "generating runnable code, sequentially merges them into an initial solution, "
            "then runs data usage and data leakage checks. "
            "Use this when you need to create an initial ML solution from scratch."
        ),
        "system_prompt": prompts["initialization"],
        "tools": [web_search, execute_code, save_code_artifact, get_sandbox_budget, read_data_files],
        "model": model,
    }

    refinement_subagent = {
        "name": "refinement",
        "description": (
            "Phase 2 of MLE-STAR: runs ablation studies to identify high-impact code blocks, "
            "then iteratively refines them through plan-code-execute cycles. Runs T outer "
            "iterations (ablation->extract) and K inner iterations (plan->code->execute) "
            "per solution. Use this when you have an initial solution needing targeted improvement."
        ),
        "system_prompt": prompts["refinement"],
        "tools": [execute_code, save_code_artifact, get_sandbox_budget, read_data_files],
        "model": model,
    }

    ensemble_subagent = {
        "name": "ensemble",
        "description": (
            "Phase 3 of MLE-STAR: takes multiple refined solutions and creates ensemble "
            "combinations through R iterations of strategy planning, implementation, and "
            "evaluation. Use this when you have 2+ refined solutions to combine."
        ),
        "system_prompt": prompts["ensemble"],
        "tools": [execute_code, save_code_artifact, get_sandbox_budget],
        "model": model,
    }

    submission_subagent = {
        "name": "submission",
        "description": (
            "Phase 4 of MLE-STAR: removes any subsampling from the best solution, trains "
            "on the full dataset, and generates the final submission.csv. Uses uncounted "
            "sandbox calls. Use this when the workflow is complete and you need a submission file."
        ),
        "system_prompt": prompts["submission"],
        "tools": [
            execute_code,
            execute_code_uncounted,
            save_code_artifact,
            get_output_file,
            get_sandbox_budget,
        ],
        "model": model,
    }

    # --- Checkpointer ---
    checkpointer = MemorySaver()

    # --- Checkpointer ---
    checkpointer = MemorySaver()

    # --- Create the root agent ---
    logger.info("[agent] Creating root deep agent with %d subagents...", 4)
    logger.info("[agent] Root tools: get_sandbox_budget, save_code_artifact, read_data_files")
    agent = create_deep_agent(
        name="mle-star",
        model=model,
        tools=[get_sandbox_budget, save_code_artifact, read_data_files],
        system_prompt=prompts["main"],
        subagents=[
            initialization_subagent,
            refinement_subagent,
            ensemble_subagent,
            submission_subagent,
        ],
        checkpointer=checkpointer,
    )

    logger.info("MLE-STAR Deep Agent created successfully")
    return agent


def configure_tools(config: MleStarConfig, work_dir: str) -> None:
    """Configure all tool modules with runtime context."""
    logger.info("[agent] Configuring tools with work_dir=%s", work_dir)

    sandbox_cfg = config.sandbox
    llm_cfg = config.llm
    params = config.mle_star

    logger.info("[agent] Configuring execute_code tool (sandbox_mode=%s)", sandbox_cfg.mode)
    ec_mod.configure({
        "work_dir": work_dir,
        "sandbox_mode": sandbox_cfg.mode,
        "sandbox_url": sandbox_cfg.url,
        "num_retries": sandbox_cfg.num_retries,
        "max_file_size_mb": sandbox_cfg.max_file_size_mb,
        "timeout": sandbox_cfg.timeout,
        "max_calls": sandbox_cfg.max_calls,
        "python_path": sandbox_cfg.python_path,
        "max_debug_attempts": params.max_debug_attempts,
        "llm_provider": llm_cfg.provider,
        "llm_model": llm_cfg.model,
        "llm_api_key": llm_cfg.get_api_key(),
        "llm_temperature": llm_cfg.temperature,
        "llm_max_tokens": llm_cfg.max_tokens,
        "azure_endpoint": llm_cfg.azure_endpoint,
        "azure_api_version": llm_cfg.azure_api_version,
    })

    logger.info("[agent] Configuring web_search tool (base_url=%s)", config.web_search.base_url)
    ws_mod.configure({
        "base_url": config.web_search.base_url,
        "num_results": config.web_search.num_results,
    })

    logger.info("[agent] Configuring file_utils tool")
    fu_mod.configure({
        "work_dir": work_dir,
    })

    logger.info("[agent] All tools configured successfully")
