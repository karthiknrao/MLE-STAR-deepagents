"""Configuration models for MLE-STAR Deep Agents edition."""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration. Supports Azure OpenAI and OpenRouter."""
    provider: Literal["azure", "openrouter"] = "openrouter"
    model: str = "deepseek/deepseek-v3.2"
    temperature: float = 1.0
    max_tokens: int = 32768
    api_key: str = ""

    # Azure-specific
    azure_endpoint: str = "https://sfc-ml-sweden.openai.azure.com/"
    azure_api_version: str = "2025-04-01-preview"

    def get_api_key(self) -> str:
        if self.provider == "azure":
            return self.api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        else:
            return self.api_key or os.environ.get("OPENROUTER_API_KEY", "")


class SandboxConfig(BaseModel):
    """Sandbox execution configuration.

    mode: "remote" — execute via HTTP sandbox API
          "local"  — execute via subprocess in a local folder
    """
    mode: Literal["remote", "local"] = "local"
    # Remote sandbox settings
    url: str = "http://autoscale-yite:8080/run_code"
    num_retries: int = 3
    max_file_size_mb: int = 2000
    timeout: int = 1800
    max_calls: int = 100
    # Local execution settings
    python_path: str = "python3"  # Path to Python interpreter for local mode
    # Shared
    work_dir: Optional[str] = None  # Set at runtime; null = auto-detect


class WebSearchConfig(BaseModel):
    """Web search configuration (SearXNG)."""
    enabled: bool = True
    base_url: str = "http://localhost:8888"
    num_results: int = 10


class ExecConfig(BaseModel):
    """Execution settings."""
    timeout: int = 1800
    device: str = "0"
    python: str = "python3"


class MLEStarParams(BaseModel):
    """MLE-STAR paper parameters (arXiv:2506.15692)."""
    num_models_to_retrieve: int = Field(default=5, description="M - models to retrieve via web search")
    max_merge_iterations: int = 3
    num_parallel_solutions: int = Field(default=2, description="L - parallel refinement solutions")
    max_ablation_iterations: int = Field(default=3, description="T - outer loop iterations")
    max_refinement_iterations: int = Field(default=5, description="K - inner loop iterations")
    max_ensemble_iterations: int = Field(default=3, description="R - ensemble iterations")
    max_debug_attempts: int = 3
    max_train_samples: int = 30000
    max_time: int = 10800


class MleStarConfig(BaseModel):
    """Top-level configuration, loaded from config.yaml."""
    mode: str = "mle-star"
    task: Optional[str] = None
    data_dir: Optional[str] = None
    log_dir: str = "./logs"
    workspace_dir: str = "./workspaces"
    log_level: str = "INFO"

    exec: ExecConfig = ExecConfig()
    sandbox: SandboxConfig = SandboxConfig()
    llm: LLMConfig = LLMConfig()
    web_search: WebSearchConfig = WebSearchConfig()
    mle_star: MLEStarParams = MLEStarParams()

    @classmethod
    def from_yaml(cls, path: str) -> "MleStarConfig":
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
