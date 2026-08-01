# services/llm_provider.py
import sys
from pathlib import Path
from typing import Any, Dict
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from openai import AsyncOpenAI
import httpx

# [REMOVED] ConversationManager 类不再需要，因为对话状态由调用方管理。

class LLMProvider:
    def __init__(self):
        self.api_url = Config.LLM_API
        self.api_key = Config.LLM_KEY
        self.timeout = 30.0
        self._client = httpx.AsyncClient()
        self._openai_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        out: Dict[str, Any] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached_tokens = getattr(prompt_details, "cached_tokens", None)
            if cached_tokens is not None:
                out["cached_tokens"] = cached_tokens

        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                out["reasoning_tokens"] = reasoning_tokens

        return {k: v for k, v in out.items() if v is not None}

    async def generate_with_metadata(
        self,
        messages: list,
        diversity_level: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        try:
            resp = await self._openai_client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=messages,
                stream=False,
                temperature=0.7,  # [MODIFIED] 固定温度，或者可以根据 diversity_level 调整
                max_tokens=2048,  # [MODIFIED] 增加 max_tokens
            )
            
            response_content = resp.choices[0].message.content
            return {
                "content": response_content,
                "usage": self._usage_to_dict(getattr(resp, "usage", None)),
                "model": getattr(resp, "model", None),
            }
            
        except httpx.HTTPStatusError as e:
            # 这里的异常类型可能是 openai.APIError 的子类，取决于 client 的实现
            raise ValueError(f"API请求失败: {e.response.text}") from e
        except Exception as e:
            # 捕获更广泛的错误
            raise RuntimeError(f"与 LLM API 交互时发生未知错误: {e}") from e

    async def generate(
        self,
        messages: list,
        diversity_level: str = "medium",
        **kwargs
    ) -> str: # [MODIFIED] Return type is now str
        result = await self.generate_with_metadata(
            messages=messages,
            diversity_level=diversity_level,
            **kwargs,
        )
        return str(result.get("content", ""))


    async def close(self):
        await self._client.aclose()
        await self._openai_client.close()


class MockLLMProvider(LLMProvider):
    async def generate_with_metadata(
        self,
        messages: list,
        diversity_level: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        print(f"Mock LLM Provider: {messages}")
        return {"content": "", "usage": {}}

    async def generate(
        self,
        messages: list, 
        diversity_level: str = "medium",
        **kwargs
    ) -> str:
        print(f"Mock LLM Provider: {messages}")
        return ""



import os

# 清除所有代理设置
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
# 创建一个单例供全局使用
llm_provider = LLMProvider()
