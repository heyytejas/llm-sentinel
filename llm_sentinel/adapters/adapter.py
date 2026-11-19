"""
Target LLM Adapter Module for LLM Sentinel.

Provides a unified interface `send(prompt, system_prompt) -> response_text`
for querying target LLM models across different providers (OpenAI, Anthropic, Ollama)
and a MockTarget for cost-free testing.
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv
import httpx

# Load environment variables from .env file
load_dotenv()


class BaseAdapter(ABC):
    """Abstract base class for all target LLM adapters."""

    def __init__(self, model: str, timeout: float = 30.0) -> None:
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def send(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt and optional system prompt to the target LLM.

        Args:
            prompt: The user prompt / adversarial probe.
            system_prompt: Optional system / developer instructions.

        Returns:
            The model's text response.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI and OpenAI-compatible endpoints (vLLM, Groq, LiteLLM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        super().__init__(model=resolved_model, timeout=timeout)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key missing. Set OPENAI_API_KEY in .env or pass api_key to OpenAIAdapter."
            )

    def send(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"] or ""


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        resolved_model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        super().__init__(model=resolved_model, timeout=timeout)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")).rstrip("/")
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "Anthropic API key missing. Set ANTHROPIC_API_KEY in .env or pass api_key to AnthropicAdapter."
            )

    def send(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content_blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")


class OllamaAdapter(BaseAdapter):
    """Adapter for locally hosted Ollama instances."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        resolved_model = model or os.getenv("OLLAMA_MODEL", "llama3")
        super().__init__(model=resolved_model, timeout=timeout)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def send(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "") or ""


class MockTarget(BaseAdapter):
    """
    Mock adapter returning deterministic or canned responses.
    Useful for testing, offline evaluation, and zero-cost CI/CD pipelines.
    """

    def __init__(
        self,
        default_response: str = "I am a helpful assistant. I cannot assist with harmful activities.",
        canned_responses: Optional[Dict[str, str]] = None,
        response_handler: Optional[Callable[[str, Optional[str]], str]] = None,
        model: str = "mock-sentinel-v1",
    ) -> None:
        super().__init__(model=model)
        self.default_response = default_response
        self.canned_responses = canned_responses or {}
        self.response_handler = response_handler
        self.history: List[Dict[str, Optional[str]]] = []

    def set_canned_response(self, prompt_keyword: str, canned_output: str) -> None:
        """Add or update a keyword-triggered canned response."""
        self.canned_responses[prompt_keyword.lower()] = canned_output

    def send(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Record interaction history for inspection
        self.history.append({"prompt": prompt, "system_prompt": system_prompt})

        # 1. Custom callback handler takes precedence if configured
        if self.response_handler:
            return self.response_handler(prompt, system_prompt)

        # 2. Check keyword triggers in canned responses
        prompt_lower = prompt.lower()
        for keyword, response in self.canned_responses.items():
            if keyword.lower() in prompt_lower:
                return response

        # 3. Fallback to default response
        return self.default_response


def get_adapter(provider: str, **kwargs: Any) -> BaseAdapter:
    """
    Factory helper to instantiate an adapter by provider name.

    Supported providers:
        - 'openai'
        - 'anthropic'
        - 'ollama'
        - 'mock'
    """
    provider_clean = provider.strip().lower()
    if provider_clean == "openai":
        return OpenAIAdapter(**kwargs)
    elif provider_clean == "anthropic":
        return AnthropicAdapter(**kwargs)
    elif provider_clean == "ollama":
        return OllamaAdapter(**kwargs)
    elif provider_clean in ("mock", "mocktarget"):
        return MockTarget(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose from: 'openai', 'anthropic', 'ollama', 'mock'."
        )
