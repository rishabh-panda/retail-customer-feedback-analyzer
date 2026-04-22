import os
import time
from typing import Optional
from functools import wraps

from requests import RequestException
from openai import OpenAIError

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError(
        "Missing required module 'langchain_openai'. "
        "Install with: pip install langchain-openai"
    )


def rate_limiter(max_calls_per_minute: int = 3):
    """
    Decorator to enforce rate limiting on API calls.
    
    Args:
        max_calls_per_minute (int): Maximum number of calls allowed per minute.
    """
    min_interval = 60.0 / max_calls_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator


class RateLimitedChatOpenAI(ChatOpenAI):
    """Wrapper for ChatOpenAI with built-in rate limiting."""
    
    @rate_limiter(max_calls_per_minute=3)
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)


def get_free_llm(temperature: float = 0.0, model_name: Optional[str] = None):
    """
    Returns a configured LLM instance based on the active provider.
    
    Supports:
        - groq  (free tier: 14,400 requests/day)
        - zhipu (free tier: GLM-4.7-Flash, rate limited to ~3 calls/min)
    
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
    temperature = max(0.0, min(1.0, float(temperature)))
    
    try:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError(
                    "GROQ_API_KEY is missing or empty. "
                    "Please set it in your .env file."
                )
            
            return RateLimitedChatOpenAI(
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
            
            return RateLimitedChatOpenAI(
                model=model_name or "glm-4.7-flash",
                temperature=temperature,
                openai_api_key=api_key.strip(),
                openai_api_base="https://open.bigmodel.cn/api/paas/v4",
                max_retries=5,
                request_timeout=90,
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