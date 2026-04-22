import os
from typing import Optional

from requests import RequestException
from openai import OpenAIError

# Attempt to import with fallback for different environments
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError(
        "Missing required module 'langchain_openai'. "
        "Install with: pip install langchain-openai"
    )


def get_free_llm(temperature: float = 0.0, model_name: Optional[str] = None):
    """
    Returns a configured LLM instance based on the active provider.

    Supports:
        - groq  (free tier: 14,400 requests/day)
        - zhipu (free tier: GLM-4.7-Flash, no token limits)

    Environment variables required:
        - ACTIVE_PROVIDER (default: 'groq')
        - GROQ_API_KEY or ZHIPU_API_KEY (depending on provider)

    Args:
        temperature (float): Sampling temperature (0.0 = deterministic).
        model_name (Optional[str]): Override default model for the provider.

    Returns:
        ChatOpenAI: Configured LLM instance.

    Raises:
        ValueError: If API key is missing or provider is unsupported.
        RequestException: If network-related errors occur.
        OpenAIError: If OpenAI API wrapper encounters an issue.
    """
    provider = os.getenv("ACTIVE_PROVIDER", "groq").strip().lower()
    temperature = max(0.0, min(1.0, float(temperature)))  # clamp between 0-1

    try:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError(
                    "GROQ_API_KEY is missing or empty. "
                    "Please set it in your .env file."
                )

            return ChatOpenAI(
                model=model_name or "llama3-70b-8192",
                temperature=temperature,
                openai_api_key=api_key.strip(),
                openai_api_base="https://api.groq.com/openai/v1",
                max_retries=3,
                request_timeout=60,
            )

        elif provider == "zhipu":
            api_key = os.getenv("ZHIPU_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError(
                    "ZHIPU_API_KEY is missing or empty. "
                    "Please set it in your .env file."
                )

            return ChatOpenAI(
                model=model_name or "glm-4.7-flash",
                temperature=temperature,
                openai_api_key=api_key.strip(),
                openai_api_base="https://open.bigmodel.cn/api/paas/v4",
                max_retries=3,
                request_timeout=60,
            )

        else:
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                "Choose either 'groq' or 'zhipu'."
            )

    except (ConnectionError, TimeoutError, RequestException) as net_err:
        raise RequestException(
            f"Network error while initializing LLM for provider '{provider}': {net_err}"
        ) from net_err

    except OpenAIError as openai_err:
        raise OpenAIError(
            f"OpenAI client error for provider '{provider}': {openai_err}"
        ) from openai_err

    except Exception as unexpected_err:
        raise RuntimeError(
            f"Unexpected failure initializing LLM for provider '{provider}': {unexpected_err}"
        ) from unexpected_err