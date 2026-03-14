import os
from langchain_openai import ChatOpenAI
from typing import Optional

def get_free_llm(temperature: float = 0, model_name: Optional[str] = None):
    """
    Factory function to get free LLM based on ACTIVE_PROVIDER env var.
    Verified free providers only [citation:1][citation:8][citation:10].
    """
    provider = os.getenv("ACTIVE_PROVIDER", "groq").lower()
    
    if provider == "groq":
        # Groq: 14,400 free requests/day [citation:1]
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        
        return ChatOpenAI(
            model=model_name or "llama3-70b-8192",  # Free tier model
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            max_retries=2,
        )
    
    elif provider == "zhipu":
        # Zhipu GLM-4.7-Flash: officially free, no token limits [citation:10]
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not found in .env")
        
        return ChatOpenAI(
            model=model_name or "glm-4.7-flash",
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://open.bigmodel.cn/api/paas/v4",
            max_retries=2,
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'zhipu'")