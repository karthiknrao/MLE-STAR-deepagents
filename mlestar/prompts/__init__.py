"""MLE-STAR prompt loader."""
from pathlib import Path


def load_prompts() -> dict[str, str]:
    """Load all prompt markdown files from the prompts/ directory."""
    prompts_dir = Path(__file__).parent
    return {
        "main": (prompts_dir / "main.md").read_text(),
        "initialization": (prompts_dir / "initialization.md").read_text(),
        "refinement": (prompts_dir / "refinement.md").read_text(),
        "ensemble": (prompts_dir / "ensemble.md").read_text(),
        "submission": (prompts_dir / "submission.md").read_text(),
    }
