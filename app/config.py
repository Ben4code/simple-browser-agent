"""Environment + LLM configuration and run caps."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    # Any OpenAI-compatible endpoint (default: OpenCode inference gateway)
    llm_api_key: str = Field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    llm_base_url: str = Field(
        default_factory=lambda: os.environ.get(
            "LLM_BASE_URL", "https://opencode.ai/inference/v1"
        )
    )
    llm_model: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "glm-5.3-flash-free")
    )

    # LLM behavior
    temperature: float = Field(
        default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.2"))
    )
    max_llm_retries: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_LLM_RETRIES", "4"))
    )

    # Run safety caps
    max_browser_calls: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_BROWSER_CALLS", "120"))
    )
    cli_timeout_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("CLI_TIMEOUT_SECONDS", "60"))
    )
    run_timeout_seconds: int = Field(
        default_factory=lambda: int(os.environ.get("RUN_TIMEOUT_SECONDS", "600"))
    )
    scout_max_iter: int = Field(
        default_factory=lambda: int(os.environ.get("SCOUT_MAX_ITER", "8"))
    )
    operator_max_iter: int = Field(
        default_factory=lambda: int(os.environ.get("OPERATOR_MAX_ITER", "25"))
    )
    scout_token_limit: int = Field(
        default_factory=lambda: int(os.environ.get("SCOUT_TOKEN_LIMIT", "4000"))
    )

    # Storage
    db_path: str = Field(
        default_factory=lambda: os.environ.get("DB_PATH", "browseragent.db")
    )

    def llm_kwargs(self) -> dict:
        """Kwargs for crewai.LLM (LiteLLM `openai/` provider prefix routes to
        any OpenAI-compatible base_url, which Z.ai serves).

        Note: no num_retries — CrewAI's OpenAI provider currently forwards it
        to completions.create() where it's an invalid kwarg. Retries are
        handled at the run layer instead.
        """
        return {
            "model": f"openai/{self.llm_model}",
            "base_url": self.llm_base_url,
            "api_key": self.llm_api_key,
            "temperature": self.temperature,
        }


settings = Settings()
