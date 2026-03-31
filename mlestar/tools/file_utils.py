"""File utility tools for MLE-STAR."""

import os
import logging
from pathlib import Path
from langchain.tools import tool

logger = logging.getLogger("mle-star.tools.file_utils")

# Module-level state
_state = {
    "work_dir": "",
}


def configure(config: dict) -> None:
    """Initialize with runtime context."""
    _state.update(config)
    logger.info(f"[file_utils] Configured with work_dir={_state['work_dir']}")


@tool
def read_data_files() -> str:
    """Read and describe all files in the input data directory.

    Returns a summary of available data files with their sizes and first few lines.
    """
    logger.info("[read_data_files] Called — scanning input directory")
    work_dir = Path(_state["work_dir"])
    input_dir = work_dir / "input"

    if not input_dir.exists():
        logger.warning("[read_data_files] No input directory found at %s", input_dir)
        return "No input directory found."

    lines = ["Input data files:"]
    file_count = 0
    for fpath in sorted(input_dir.iterdir()):
        if fpath.is_file():
            file_count += 1
            size_kb = fpath.stat().st_size / 1024
            lines.append(f"  {fpath.name} ({size_kb:.1f} KB)")
            logger.info("[read_data_files]   Found file: %s (%.1f KB)", fpath.name, size_kb)
            # Show header for CSV files
            if fpath.suffix == ".csv":
                try:
                    with open(fpath, "r") as f:
                        header = f.readline().strip()
                    lines.append(f"    Columns: {header[:200]}")
                    logger.info("[read_data_files]   Columns: %s", header[:200])
                except Exception:
                    pass

    logger.info("[read_data_files] Found %d files in input directory", file_count)
    return "\n".join(lines)


@tool
def get_output_file(filename: str = "submission.csv") -> str:
    """Check for an output file in the workspace.

    Args:
        filename: The filename to look for (default: "submission.csv").

    Returns:
        The file path if found, or an error message.
    """
    logger.info("[get_output_file] Looking for: %s", filename)
    work_dir = Path(_state["work_dir"])

    possible_paths = [
        work_dir / "final" / filename,
        work_dir / filename,
    ]

    for path in possible_paths:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            logger.info("[get_output_file] Found: %s (%.1f KB)", path, size_kb)
            return f"Found: {path} ({size_kb:.1f} KB)"

    logger.warning("[get_output_file] File not found: %s", filename)
    return f"File not found: {filename}"
