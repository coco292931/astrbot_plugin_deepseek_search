import logging
import os
import urllib.parse

import httpx

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_deepseek_search"

SEARCH_MAX_RESULTS = 8          # 本地截断上限（dsh: searchMaxResults）
SEARCH_TIMEOUT_MS = 30000       # dsh: searchTimeoutMs
SEARCH_TIMEOUT_S = SEARCH_TIMEOUT_MS / 1000

# ---- dsh web-search-deepseek provider 常量（照抄 @deepseek-ai/dsh-web-search-deepseek）----
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic/v1"  # /messages 由调用方拼接
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_DEFAULT_API_VERSION = "2023-06-01"
DEEPSEEK_DEFAULT_MAX_TOKENS = 4096
DEEPSEEK_DEFAULT_MAX_USES = 5
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_USER_AGENT = "deepseek-harness/0.0.1"


@register(PLUGIN_NAME, "coco292931",
          "让 AstrBot LLM 接入 DeepSeek 官方联网搜索（web_search，dsh 方案）",
          "0.3.1",
          "https://github.com/coco292931/astrbot_plugin_deepseek_search")
class DeepSeekSearchPlugin(Star):
    """接入 DeepSeek 官方联网搜索（web_search）的 AstrBot 插件。

    调用方式已切换为 dsh 实测验证的方案：直连 DeepSeek 的 Anthropic 兼容
    Messages API（https://api.deepseek.com/anthropic/v1/messages），请求体声明
    原生服务端工具 tools=[{"type":"web_search_20250305","name":"web_search",...}]，
    能真实触发 DeepSeek 原生联网搜索并返回结构化结果块。

    - 响应解析与 dsh 源码（@deepseek-ai/dsh-web-search-deepseek）保持一致：
      web_search_tool_result 块内的 web_search_result 条目作为来源；
      snippet 取自 text 块的 citations[]（url→cited_text，首次出现优先）；
      encrypted_content 不解密、不使用（dsh 源码同样只认 cited_text）。
    - strict 模式：响应里没有 web_search_tool_result 块直接报错，不做降级。
    - API Key：优先读配置项 api_key，其次环境变量 DEEPSEEK_API_KEY。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.logger = logging.getLogger("astrbot")
        self.api_key_override = (config.get("api_key") or "").strip()
        self.api_base = (config.get("api_base") or "").strip() or DEEPSEEK_DEFAULT_BASE_URL
        self.model = (config.get("model") or "").strip() or DEEPSEEK_DEFAULT_MODEL

    # ---------- API Key ----------

    def _resolve_api_key(self) -> str:
        """配置项 api_key 优先，其次环境变量 DEEPSEEK_API_KEY。"""
        if self.api_key_override:
            return self.api_key_override
        env_key = (os.environ.get(DEEPSEEK_KEY_ENV) or "").strip()
        if env_key:
            return env_key
        raise RuntimeError(
            "未找到 DeepSeek API Key：配置项 api_key 为空，且环境变量 "
            f"{DEEPSEEK_KEY_ENV} 为空。"
        )

    # ---------- 调用（dsh 方案）----------

    async def _call_deepseek_web_search(self, query: str) -> dict:
        """直连 DeepSeek Anthropic 兼容 /messages，触发 web_search_20250305 原生搜索。

        返回 DeepSeek 原始响应 dict（content[] 里含 text / server_tool_use /
        web_search_tool_result 等 block）。
        """
        key = self._resolve_api_key()
        endpoint = self.api_base.rstrip("/") + "/messages"
        payload = {
            "model": self.model,
            "max_tokens": DEEPSEEK_DEFAULT_MAX_TOKENS,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"Perform a web search for the query: {query}"
                }]
            }],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": DEEPSEEK_DEFAULT_MAX_USES,
            }],
        }
        headers = {
            "x-api-key": key,
            "authorization": f"Bearer {key}",
            "anthropic-version": DEEPSEEK_DEFAULT_API_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": DEEPSEEK_USER_AGENT,
        }
        self.logger.info(
            f"[deepseek_search] dsh 方案调用 Anthropic Messages: {endpoint} model={self.model}"
        )
        async with httpx.AsyncClient(
            timeout=SEARCH_TIMEOUT_S, follow_redirects=False
        ) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code >= 300:  # 3xx 重定向同样视为错误（dsh: redirect: "error"）
                raise RuntimeError(self._format_api_error(resp))
            data = resp.json()
        return data

    @staticmethod
    def _format_api_error(resp: httpx.Response) -> str:
        """按 dsh 解析错误体：error 字符串 / error.message / message。"""
        try:
            parsed = resp.json()
        except Exception:
            return f"DeepSeek API error (HTTP {resp.status_code}): {resp.text[:500]}"
        detail = None
        error = parsed.get("error")
        if isinstance(error, str):
            detail = error
        elif isinstance(error, dict):
            detail = error.get("message")
        if not detail:
            detail = parsed.get("message")
        if detail:
            return f"DeepSeek API error (HTTP {resp.status_code}): {detail}"
        return f"DeepSeek API error (HTTP {resp.status_code}): {resp.text[:500]}"

    # ---------- dsh 响应解析 ----------

    @staticmethod
    def _blocks(data: dict) -> list:
        return data.get("content") or []

    @staticmethod
    def _extract_content(blocks: list) -> str:
        """把 text 块的正文拼起来作为 provider answer（dsh 输出格式里的 content）。"""
        parts = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = (block.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)

    @classmethod
    def _citation_snippets(cls, blocks: list) -> dict:
        """dsh citationSnippets：text 块 citations[] 里 url→cited_text（首次出现优先）。

        encrypted_content 不解密也不使用——dsh 源码同样只认 cited_text。
        """
        mapping = {}
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            for cite in block.get("citations") or []:
                if not isinstance(cite, dict):
                    continue
                url = (cite.get("url") or "").strip()
                cited_text = (cite.get("cited_text") or "").strip()
                if url and cited_text and url not in mapping:
                    mapping[url] = cited_text
        return mapping

    def _map_anthropic_response(self, data: dict) -> list:
        """dsh mapAnthropicResponse：strict 模式 + 按 url 去重。

        遍历 web_search_tool_result 块的 web_search_result 条目，用 text 块
        citations 的 cited_text 拼 snippet；title←title、publishedAt←page_age。
        没有 web_search_tool_result 块时报错（strict，不降级）。
        """
        blocks = self._blocks(data)
        result_blocks = [
            b for b in blocks
            if isinstance(b, dict) and b.get("type") == "web_search_tool_result"
        ]
        if not result_blocks:
            raise RuntimeError(
                "DeepSeek returned no web_search_tool_result blocks; "
                "the request may not have triggered native web search"
            )
        snippets = self._citation_snippets(blocks)
        seen = set()
        sources = []
        for block in result_blocks:
            for item in block.get("content") or []:
                if not isinstance(item, dict) or item.get("type") != "web_search_result":
                    continue
                url = (item.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                source = {"url": url}
                title = (item.get("title") or "").strip()
                if title:
                    source["title"] = title
                snippet = (snippets.get(url) or "").strip()
                if snippet:
                    source["snippet"] = snippet
                page_age = (item.get("page_age") or "").strip()
                if page_age:
                    source["publishedAt"] = page_age
                sources.append(source)
        return sources

    # ---------- 输出格式（dsh 规范）----------

    @staticmethod
    def _format_label(source: dict) -> str:
        """label = title，无 title 则用 url 的 hostname。"""
        if source.get("title"):
            return source["title"]
        host = urllib.parse.urlparse(source["url"]).netloc
        return host if host else source["url"]

    @staticmethod
    def _format_sources_section(sources: list) -> str:
        """按 dsh formatSearchOutput 生成 Sources 列表块。"""
        lines = []
        for s in sources:
            label = DeepSeekSearchPlugin._format_label(s)
            url = s["url"]
            meta = []
            if s.get("snippet"):
                meta.append(s["snippet"])
            if s.get("publishedAt"):
                meta.append(f"({s['publishedAt']})")
            suffix = f" — {' '.join(meta)}" if meta else ""
            lines.append(f"- [{label}]({url}){suffix}")
        return "Sources:\n" + "\n".join(lines)

    @staticmethod
    def _build_llm_text(content: str, sources: list, truncated: bool) -> str:
        """按 dsh formatSearchOutput 拼装：content + 空行 + Sources + 截断提示 + cite 指令。"""
        parts = []
        if content:
            parts.append(content.strip())
        if sources:
            parts.append(DeepSeekSearchPlugin._format_sources_section(sources))
        elif not content:
            parts.append("No results found.")
        if truncated:
            parts.append(f"(Showing the first {len(sources)} sources. Refine the query for more.)")
        parts.append("Cite the relevant URLs above as markdown links in your answer.")
        return "\n\n".join(parts)

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
                "content": "No results found.\n\nCite the relevant URLs above as markdown links in your answer.",
                "sources": [],
                "truncated": False,
                "error": "缺少必填参数 query",
            }
        try:
            data = await self._call_deepseek_web_search(query)
            content = self._extract_content(self._blocks(data))
            all_sources = self._map_anthropic_response(data)
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
                "content": f"web_search 调用失败: {e}\n\nCite the relevant URLs above as markdown links in your answer.",
                "sources": [],
                "truncated": False,
                "error": str(e),
            }
