"""LLM factory. Default is OpenAI; swap provider with LLM_PROVIDER in .env
without touching the chain (the 'swap models' concept from the brief)."""
import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.0):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                          temperature=temperature)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
