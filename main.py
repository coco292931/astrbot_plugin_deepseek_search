import json
import logging
import os
import urllib.error
import urllib.request

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_deepseek_search"


@register(PLUGIN_NAME, "coco292931",
          "让 AstrBot LLM 接入 DeepSeek 官方搜索工具",
          "0.1.0",
          "https://github.com/coco292931/astrbot_plugin_deepseek_search")
class DeepSeekSearchPlugin(Star):
    """接入 DeepSeek 官方联网搜索（web_search）的 AstrBot 插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.logger = logging.getLogger("astrbot")
        self.api_key = (os.environ.get("DEEPSEEK_API_KEY")
                        or config.get("api_key", "") or "").strip()
        self.base_url = (config.get("base_url", "https://api.deepseek.com")
                         .rstrip("/"))
        self.model = config.get("model", "deepseek-chat")

    def _chat_with_web_search(self, query: str, timeout: int = 60) -> dict:
        """调用 DeepSeek 官方 web_search（同步 HTTP，零第三方依赖）。"""
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY（环境变量或插件配置 api_key）")

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": query}],
            "tools": [{"type": "web_search", "web_search": {"enable": True}}],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"DeepSeek API HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 请求失败: {e}") from e

    @staticmethod
    def _extract_citations(msg: dict) -> list:
        """兼容不同字段名的引用来源提取。"""
        out = []
        for key in ("annotations", "citations", "source_documents"):
            for item in (msg.get(key) or []):
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(item.get("url") or item.get("title")
                               or item.get("text") or "")
        return [x for x in out if x]

    @filter.llm_tool(name="deepseek_web_search")
    async def deepseek_web_search(self, event: AstrMessageEvent, query: str = "") -> dict:
        """使用 DeepSeek 官方联网搜索检索实时信息，并返回带引用来源的搜索结果。

        Args:
            query(string): 必填。要搜索的问题或关键词，例如"2025年苹果发布会发布了什么新品？"。
        """
        query = (query or "").strip()
        if not query:
            return {"status": "error",
                    "message": "缺少必填参数 query：需要提供要搜索的问题或关键词。"}
        try:
            resp = self._chat_with_web_search(query)
            if not (resp.get("choices") and resp["choices"]):
                return {"status": "error", "message": "DeepSeek 返回空结果", "data": resp}
            msg = resp["choices"][0].get("message", {})
            content = msg.get("content") or ""
            citations = self._extract_citations(msg)
            usage = resp.get("usage", {})
            answer = content
            # 引用标记对 LLM 阅读更友好：在正文尾部补充来源列表
            if citations:
                refs = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(citations))
                answer = f"{content}\n\n引用来源：\n{refs}"
            return {
                "status": "success",
                "message": f"搜索结果完成，正文 {len(content)} 字，引用 {len(citations)} 条",
                "data": {
                    "query": query,
                    "answer": answer,
                    "citations": citations,
                    "model": resp.get("model", self.model),
                    "usage": usage,
                },
            }
        except Exception as e:
            self.logger.error(f"deepseek_web_search 调用失败: {e}")
            return {"status": "error", "message": str(e)}
