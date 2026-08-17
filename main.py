import json
import logging
import urllib.parse
from typing import Any, Optional

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_deepseek_search"

SEARCH_MAX_RESULTS = 8          # dsh: searchMaxResults
SEARCH_TIMEOUT_MS = 30000       # dsh: searchTimeoutMs
SEARCH_TIMEOUT_S = SEARCH_TIMEOUT_MS / 1000


@register(PLUGIN_NAME, "coco292931",
          "让 AstrBot LLM 接入 DeepSeek 官方联网搜索（web_search，provider 模式）",
          "0.2.0",
          "https://github.com/coco292931/astrbot_plugin_deepseek_search")
class DeepSeekSearchPlugin(Star):
    """接入 DeepSeek 官方联网搜索（web_search）的 AstrBot 插件。

    模型调用走 AstrBot provider 机制，不手配 API Key / Base URL / Model。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.logger = logging.getLogger("astrbot")
        self.llm_provider_id = config.get(
            "llm_provider_id", "deepseek/deepseek-v4-flash"
        ).strip() or "deepseek/deepseek-v4-flash"

    # ---------- provider 调用 ----------

    async def _call_provider(self, query: str) -> Any:
        """通过 AstrBot provider 发起搜索（模型侧触发 DeepSeek web_search）。"""
        provider = self.context.get_provider_by_id(self.llm_provider_id)
        if not provider:
            raise RuntimeError(
                f"未找到 llm_provider_id='{self.llm_provider_id}' 对应的 Provider，"
                "请检查插件配置或 AstrBot 已配置的 provider。"
            )
        resp = await provider.text_chat(
            prompt=query,
            persist=False,
            max_tokens=1024,
        )
        if resp is None:
            raise RuntimeError("Provider 返回空响应。")
        return resp

    # ---------- dsh 结果解析 ----------

    @staticmethod
    def _extract_message_dict(raw: Any) -> dict:
        """从 raw_completion 中尽量提取 message 字典，兼容 pydantic 对象与 dict。"""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            msg = raw.get("choices", [{}])[0].get("message", {}) if raw.get("choices") else {}
            return msg if isinstance(msg, dict) else {}
        # pydantic / 普通对象
        try:
            if hasattr(raw, "model_dump"):
                data = raw.model_dump()
                if isinstance(data, dict):
                    choices = data.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        return choices[0].get("message") or {}
            if hasattr(raw, "choices") and raw.choices:
                choice = raw.choices[0]
                if hasattr(choice, "message"):
                    msg = choice.message
                    if hasattr(msg, "model_dump"):
                        return msg.model_dump()
        except Exception:
            pass
        return {}

    @staticmethod
    def _iter_citation_items(msg: dict):
        """兼容不同字段名（annotations / citations / source_documents）的引用来源。"""
        for key in ("annotations", "citations", "source_documents"):
            for item in msg.get(key) or []:
                if isinstance(item, str):
                    yield {"url": item}
                elif isinstance(item, dict):
                    yield item

    def _build_sources(self, msg: dict) -> list[dict]:
        """提取全部有效 sources（不做截断，截断判定交给调用方）。"""
        sources: list[dict] = []
        for item in self._iter_citation_items(msg):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            title = (item.get("title") or "").strip() or None
            snippet = (item.get("snippet") or item.get("text") or "").strip() or None
            published_at = (item.get("publishedAt") or item.get("published_at")
                            or "").strip() or None
            sources.append({
                "url": url,
                "title": title,
                "snippet": snippet,
                "publishedAt": published_at,
            })
        return sources

    @staticmethod
    def _format_label(source: dict) -> str:
        """label = title，无 title 则用 url 的 hostname。"""
        if source.get("title"):
            return source["title"]
        host = urllib.parse.urlparse(source["url"]).netloc
        return host if host else source["url"]

    @staticmethod
    def _format_sources_section(sources: list[dict]) -> str:
        """按 dsh 规范生成 Sources 列表块。"""
        if not sources:
            return ""
        lines = ["Sources:"]
        for s in sources:
            label = DeepSeekSearchPlugin._format_label(s)
            url = s["url"]
            parts = [f"[{label}]({url})"]
            extra = []
            if s.get("snippet"):
                extra.append(s["snippet"])
            if s.get("publishedAt"):
                extra.append(s["publishedAt"])
            if extra:
                parts.append(" — ".join(extra))
            lines.append("- " + " — ".join(parts))
        return "\n".join(lines)

    def _build_llm_text(self, content: str, sources: list[dict], truncated: bool) -> str:
        """严格按 dsh 规范拼装返回给 LLM 的文本。"""
        parts: list[str] = []
        if content:
            parts.append(content.strip())
        src_block = self._format_sources_section(sources)
        if src_block:
            if content:
                parts.append("")  # 空一行
            parts.append(src_block)
        elif not content:
            parts.append("No results found.")
        if truncated:
            parts.append(f"(Showing the first {len(sources)} sources. Refine the query for more.)")
        parts.append("Cite the relevant URLs above as markdown links in your answer.")
        return "\n".join(parts)

    # ---------- 工具 ----------

    @filter.llm_tool(name="web_search")
    async def web_search(self, event: AstrMessageEvent, query: str = "") -> dict:
        """使用 DeepSeek 官方联网搜索检索实时信息，并返回带引用来源的搜索结果。

        Args:
            query(string): 必填。要搜索的问题或关键词，例如"2025年苹果发布会发布了什么新品？"。
        """
        query = (query or "").strip()
        if not query:
            return {
                "content": "No results found.\nCite the relevant URLs above as markdown links in your answer.",
                "sources": [],
                "truncated": False,
                "error": "缺少必填参数 query",
            }
        try:
            resp = await self._call_provider(query)
            if hasattr(resp, "completion_text"):
                content = (resp.completion_text or "").strip()
            else:
                content = ""
            msg = self._extract_message_dict(getattr(resp, "raw_completion", None))
            all_sources = self._build_sources(msg)
            truncated = len(all_sources) > SEARCH_MAX_RESULTS
            sources = all_sources[:SEARCH_MAX_RESULTS]
            text = self._build_llm_text(content, sources, truncated)
            return {
                "content": text,
                "sources": sources,
                "truncated": truncated,
            }
        except Exception as e:
            self.logger.error(f"web_search 调用失败: {e}")
            return {
                "content": f"web_search 调用失败: {e}\nCite the relevant URLs above as markdown links in your answer.",
                "sources": [],
                "truncated": False,
                "error": str(e),
            }
