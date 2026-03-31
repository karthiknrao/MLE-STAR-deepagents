"""Code execution tool for MLE-STAR.

Supports two modes:
- "remote": Execute via HTTP sandbox API (upload files, run remotely, download results)
- "local": Execute via subprocess in a local folder with a configurable Python environment

Both modes include call counting, score extraction, and auto-debug retry logic.
"""

import os
import re
import time
import base64
import logging
import subprocess
import requests
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain.chat_models import init_chat_model

logger = logging.getLogger("mle-star.tools.execute_code")

SCORE_PATTERN = re.compile(
    r"Final Validation Performance[:\s]+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Module-level state, initialized by configure()
_state = {
    "work_dir": "",
    # Sandbox mode: "remote" or "local"
    "sandbox_mode": "local",
    # Remote sandbox settings
    "sandbox_url": "",
    "num_retries": 3,
    "max_file_size_mb": 2000,
    "timeout": 1800,
    # Call budget
    "call_count": 0,
    "max_calls": 100,
    "max_debug_attempts": 3,
    # Local execution settings
    "python_path": "python3",
    # LLM for auto-debug
    "llm_provider": "openrouter",
    "llm_model": "deepseek/deepseek-v3.2",
    "llm_api_key": "",
    "llm_temperature": 0.3,
    "llm_max_tokens": 32768,
    "azure_endpoint": "",
    "azure_api_version": "",
}


def configure(config: dict) -> None:
    """Initialize the tool with runtime context from the agent."""
    _state.update(config)
    logger.info("[configure] Execute code tool configured:")
    logger.info("[configure]   sandbox_mode=%s", _state["sandbox_mode"])
    logger.info("[configure]   python_path=%s", _state["python_path"])
    logger.info("[configure]   work_dir=%s", _state["work_dir"])
    logger.info("[configure]   max_calls=%d, max_debug_attempts=%d", _state["max_calls"], _state["max_debug_attempts"])
    logger.info("[configure]   timeout=%ds, num_retries=%d", _state["timeout"], _state["num_retries"])
    logger.info("[configure]   llm_provider=%s, llm_model=%s", _state["llm_provider"], _state["llm_model"])


def _extract_score(output: str) -> Optional[float]:
    match = SCORE_PATTERN.search(output)
    if match:
        try:
            score = float(match.group(1))
            logger.info("[score] Extracted validation score: %.6f", score)
            return score
        except ValueError:
            logger.warning("[score] Found score pattern but failed to parse: %s", match.group(1))
    logger.debug("[score] No 'Final Validation Performance' found in output (len=%d)", len(output))
    return None


# ---------------------------------------------------------------------------
# Remote sandbox helpers
# ---------------------------------------------------------------------------

def _get_files_to_upload() -> dict[str, str]:
    """Base64-encode all files in work_dir for sandbox upload."""
    work_dir = Path(_state["work_dir"])
    max_size = _state["max_file_size_mb"] * 1024 * 1024
    encoded = {}
    logger.info("[upload] Scanning work_dir for files to upload: %s", work_dir)

    for root, dirs, files in os.walk(work_dir):
        if "backup" in root or "__pycache__" in root:
            continue
        for fname in files:
            full_path = Path(root) / fname
            try:
                rel_path = str(full_path.relative_to(work_dir))
            except ValueError:
                rel_path = fname

            try:
                if full_path.stat().st_size > max_size:
                    continue
                with open(full_path, "rb") as f:
                    encoded[rel_path] = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass

    logger.info("[upload] Encoded %d files for upload", len(encoded))
    return encoded


def _save_files_from_response(files_data: dict[str, str]) -> list[str]:
    """Save base64-encoded files from sandbox to work_dir."""
    work_dir = Path(_state["work_dir"])
    saved = []
    logger.info("[download] Saving %d files from sandbox response", len(files_data))
    for file_path, b64_content in files_data.items():
        try:
            full_path = work_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(base64.b64decode(b64_content))
            saved.append(file_path)
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {e}")
    logger.info("[download] Saved %d files: %s", len(saved), saved)
    return saved


def _run_remote(code: str) -> dict:
    """Execute code via the remote HTTP sandbox API."""
    start_time = time.time()
    logger.info("[remote] Starting remote execution (code length: %d chars)", len(code))

    setup_code = """
import os
os.makedirs("./input", exist_ok=True)
os.makedirs("./working", exist_ok=True)
os.makedirs("./final", exist_ok=True)
os.chdir(".")
"""
    full_code = setup_code + code

    files_to_upload = _get_files_to_upload()
    files_to_fetch = [
        "final/submission.csv",
        "final/prediction.csv",
        "submission.csv",
        "prediction.csv",
        "output.csv",
        "result.csv",
    ]

    payload = {
        "code": full_code,
        "language": "python",
        "files": files_to_upload,
        "fetch_files": files_to_fetch,
        "compile_timeout": _state["timeout"],
        "run_timeout": _state["timeout"],
    }

    response_data = None
    last_error = None
    for attempt in range(_state["num_retries"] + 1):
        try:
            logger.info("[remote] POST to sandbox (attempt %d/%d): %s", attempt + 1, _state["num_retries"] + 1, _state["sandbox_url"])
            resp = requests.post(
                _state["sandbox_url"],
                json=payload,
                timeout=_state["timeout"] + 120,
            )
            resp.raise_for_status()
            response_data = resp.json()
            break
        except Exception as e:
            last_error = e
            logger.warning("[remote] Sandbox request failed (attempt %d): %s", attempt + 1, e)
            if attempt < _state["num_retries"]:
                time.sleep(2**attempt)

    exec_time = time.time() - start_time

    if response_data is None:
        logger.error("[remote] All sandbox connection attempts failed: %s", last_error)
        return {
            "success": False,
            "output": "",
            "error": f"Sandbox connection failed: {last_error}",
            "exec_time": exec_time,
            "validation_score": None,
            "files_created": [],
        }

    run_result = response_data.get("run_result", {})
    stdout = run_result.get("stdout", "")
    stderr = run_result.get("stderr", "")

    saved_files = _save_files_from_response(response_data.get("files", {}))

    output = stdout
    if stderr:
        output += f"\n[STDERR]\n{stderr}"

    has_error = bool(stderr and ("Error" in stderr or "Traceback" in stderr))
    has_timeout = run_result.get("timeout", False)
    score = _extract_score(output)
    success = not has_error and not has_timeout and score is not None

    logger.info("[remote] Execution completed in %.1fs — success=%s, score=%s, has_error=%s, timeout=%s",
                exec_time, success, score, has_error, has_timeout)

    return {
        "success": success,
        "output": output,
        "error": stderr if has_error else None,
        "exec_time": round(exec_time, 2),
        "validation_score": score,
        "files_created": saved_files,
    }


# ---------------------------------------------------------------------------
# Local execution helpers
# ---------------------------------------------------------------------------

def _run_local(code: str) -> dict:
    """Execute code locally via subprocess in the work directory.

    Uses the configured python_path (e.g., a venv Python) and runs
    the script with work_dir as cwd. Files are written/read directly
    from the local filesystem — no upload/download needed.
    """
    start_time = time.time()
    work_dir = Path(_state["work_dir"])
    python = _state["python_path"]

    logger.info("[local] Starting local execution (code length: %d chars)", len(code))
    logger.info("[local] Python: %s, cwd: %s", python, work_dir)

    # Write code to a temp script in work_dir
    script_path = work_dir / "_exec.py"
    #script_path = "_exec.py"
    logger.info("[local] %s", script_path)
    script_path.write_text(code)
    cwd = os.getcwd()
    try:
        result = subprocess.run(
            [python, str(script_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_state["timeout"],
            env={**os.environ, "PYTHONPATH": str(work_dir)},
        )
        exec_time = time.time() - start_time
        output = result.stdout
        stderr = result.stderr
        has_error = result.returncode != 0
        score = _extract_score(output + stderr)

        logger.info("[local] Process exited with returncode=%d in %.1fs", result.returncode, exec_time)
        if has_error:
            logger.warning("[local] Execution error (first 500 chars): %s", stderr[:500])
        if score is not None:
            logger.info("[local] Score: %.6f", score)

        # Check for output files created during execution
        files_created = []
        for fname in ["final/submission.csv", "final/prediction.csv",
                       "submission.csv", "prediction.csv"]:
            fpath = work_dir / fname
            if fpath.exists():
                files_created.append(fname)

        return {
            "success": not has_error and score is not None,
            "output": output,
            "error": stderr if has_error else None,
            "exec_time": round(exec_time, 2),
            "validation_score": score,
            "files_created": files_created,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Execution timeout after {_state['timeout']}s",
            "exec_time": _state["timeout"],
            "validation_score": None,
            "files_created": [],
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "exec_time": round(time.time() - start_time, 2),
            "validation_score": None,
            "files_created": [],
        }


# ---------------------------------------------------------------------------
# Auto-debug (LLM-powered code fix)
# ---------------------------------------------------------------------------

def _debug_code(code: str, error: str) -> Optional[str]:
    """Use LLM to fix code errors. Returns fixed code or None."""
    try:
        logger.info("[debug] Calling LLM to fix code error (error length: %d chars)", len(error))
        api_key = _state["llm_api_key"]
        if _state["llm_provider"] == "openrouter":
            model = init_chat_model(
                model=_state["llm_model"],
                model_provider="openai",
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                temperature=0.3,
                max_tokens=_state["llm_max_tokens"],
            )
        else:
            model = init_chat_model(
                model=_state["llm_model"],
                model_provider="azure",
                azure_endpoint=_state["azure_endpoint"],
                azure_api_version=_state["azure_api_version"],
                api_key=api_key,
                temperature=0.3,
                max_tokens=_state["llm_max_tokens"],
            )

        prompt = f"""# Code with an error:
{code}

# Error:
{error}

# Your task
- Please revise the code to fix the error.
- Do not remove subsampling if it exists.
- Provide the improved, self-contained Python script again.
- All the provided input data is stored in "input" directory.
- Remember to print a line with 'Final Validation Performance: <score>'.
- The code should be a single-file Python program, self-contained, executable as-is.
- Your response should only contain a single code block.
- Do not use exit() function.
"""
        response = model.invoke(prompt)
        text = response.content

        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            fixed = match.group(1).strip()
            logger.info("[debug] LLM returned fixed code (length: %d chars)", len(fixed))
            return fixed
        if text.strip().startswith(("import ", "from ")):
            logger.info("[debug] LLM returned raw code (length: %d chars)", len(text.strip()))
            return text.strip()
        logger.warning("[debug] LLM response did not contain valid code")
        return None
    except Exception as e:
        logger.error("[debug] Debug LLM call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Core execution dispatcher
# ---------------------------------------------------------------------------

def _execute(code: str, script_name: str, counted: bool, max_debug: int) -> dict:
    """Shared execution logic with call counting and auto-debug."""
    logger.info("[execute] %s script '%s' (counted=%s, max_debug=%d)",
                "Counted" if counted else "Uncounted", script_name, counted, max_debug)
    if counted:
        if _state["call_count"] >= _state["max_calls"]:
            logger.error("[execute] Call budget exhausted: %d/%d", _state["call_count"], _state["max_calls"])
            return {
                "success": False,
                "output": "",
                "error": f"Sandbox call limit reached ({_state['max_calls']} calls)",
                "exec_time": 0.0,
                "validation_score": None,
                "files_created": [],
            }
        _state["call_count"] += 1
        logger.info(f"Execution call {_state['call_count']}/{_state['max_calls']}")

    # Ensure directories exist
    work_dir = Path(_state["work_dir"])
    (work_dir / "input").mkdir(parents=True, exist_ok=True)
    (work_dir / "final").mkdir(parents=True, exist_ok=True)

    current_code = code
    for attempt in range(max_debug + 1):
        # Route to remote or local based on mode
        if _state["sandbox_mode"] == "remote":
            result = _run_remote(current_code)
        else:
            result = _run_local(current_code)

        if result["success"]:
            logger.info("[execute] Script '%s' succeeded — score=%.6f, time=%.1fs",
                        script_name, result["validation_score"] or 0, result["exec_time"])
            return result

        if result["error"] and attempt < max_debug:
            logger.info("Execution error, debugging (attempt %d/%d)...", attempt + 1, max_debug)
            logger.info("Error : %s", str(result["error"])[:500])
            fixed = _debug_code(current_code, result["error"])
            if fixed:
                current_code = fixed
                logger.info("[execute] Retrying with debugged code (length: %d chars)", len(fixed))
            else:
                logger.warning("[execute] Debug LLM could not produce fixed code, giving up")
                break
        else:
            logger.warning("[execute] Script '%s' failed after all retries", script_name)
            break

    logger.info("[execute] Final result for '%s': success=%s, score=%s",
                script_name, result["success"], result["validation_score"])
    return result


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def execute_code(code: str, script_name: str = "train.py") -> dict:
    """Execute Python code (counted against call budget).

    In "remote" mode, sends code + files to an HTTP sandbox API.
    In "local" mode, runs code via subprocess in the work directory using
    the configured Python interpreter.

    Auto-debugs on failure (up to max_debug_attempts retries).

    Args:
        code: Python code to execute.
        script_name: Name for the script (for logging only).

    Returns:
        Dict with keys: success (bool), output (str), error (str|None),
        validation_score (float|None), files_created (list[str]), exec_time (float).
    """
    max_debug = _state.get("max_debug_attempts", 3)
    return _execute(code, script_name, counted=True, max_debug=max_debug)


@tool
def execute_code_uncounted(code: str, script_name: str = "submission.py") -> dict:
    """Execute Python code WITHOUT counting against the call budget.

    Use only for final submission generation. Same behavior as execute_code
    but does not increment the call counter.
    """
    max_debug = _state.get("max_debug_attempts", 3)
    return _execute(code, script_name, counted=False, max_debug=max_debug)


@tool
def save_code_artifact(code: str, filename: str) -> str:
    """Save a code artifact to the workspace directory.

    Args:
        code: The code content to save.
        filename: Filename to save as (e.g. "initial_solution.py").

    Returns:
        The full path where the file was saved.
    """
    work_dir = Path(_state["work_dir"])
    path = work_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return f"Saved to: {path}"


@tool
def get_sandbox_budget() -> dict:
    """Check remaining execution call budget.

    Returns:
        Dict with 'used' (int), 'limit' (int), 'remaining' (int) keys.
    """
    return {
        "used": _state["call_count"],
        "limit": _state["max_calls"],
        "remaining": max(0, _state["max_calls"] - _state["call_count"]),
    }
