# astrbot_plugin_deepseek_search
让 AstrBot LLM 接入 DeepSeek 官方联网搜索工具（web_search，dsh 方案）。

## 使用
1. 安装插件。
2. 配置 API Key：优先填插件配置 `api_key`，留空则自动读取环境变量 `DEEPSEEK_API_KEY`。
3. 插件自动注册 LLM 工具 `web_search(query)`，模型在需要实时信息时自动调用。
4. 返回结果严格对齐 DeepSeek Harness（dsh）的 web_search 规范：
   正文 → 空行 → `Sources:` 列表 → 截断提示（如有）→ `Cite the relevant URLs above as markdown links in your answer.`

## 配置
- `api_key`: 可选，DeepSeek API Key，覆盖环境变量 `DEEPSEEK_API_KEY`。
- `api_base`: 可选，Anthropic 兼容 API 地址，默认 `https://api.deepseek.com/anthropic/v1`。
- `model`: 可选，模型名，默认 `deepseek-v4-flash`。

## 说明
- 调用方式为 dsh 实测验证方案：直连 `POST {api_base}/messages`，请求体声明原生服务端工具 `web_search_20250305`，能真实触发 DeepSeek 原生联网搜索。
- 响应解析对齐 dsh 源码：来源取 `web_search_tool_result` 块内 `web_search_result` 条目；snippet 取 text 块 citations 的 `cited_text`；`encrypted_content` 不解密不使用。
- strict 模式：响应无 `web_search_tool_result` 块直接报错，不降级。
- 搜索上限 8 条，超时 30 秒。
