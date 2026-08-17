# astrbot_plugin_deepseek_search
让 AstrBot LLM 接入 DeepSeek 官方联网搜索工具（web_search）。

## 使用
1. 安装插件（放入 /AstrBot/data/plugins/ 并加载）。
2. 在 AstrBot 中配置好对应的 Provider（如 deepseek），并确保模型为支持 web_search 的版本。
3. 插件配置 `llm_provider_id`（默认 `deepseek/deepseek-v4-flash`），格式为 AstrBot 的 provider_id（平台名/模型名）。
4. 插件自动注册 LLM 工具 `web_search(query)`，模型在需要实时信息时自动调用。
5. 返回结果严格对齐 DeepSeek Harness 的 web_search 规范：
   - 正文 → 空行 → `Sources:` 列表 → 截断提示（如有）→ `Cite the relevant URLs above as markdown links in your answer.`

## 配置
- `llm_provider_id`: 模型 Provider ID（平台名/模型名）。未配置时默认 `deepseek/deepseek-v4-flash`。

## 说明
- 搜索上限 8 条，超时 30 秒。
- 不再手动配置 DEEPSEEK_API_KEY / base_url / model，全部走 AstrBot provider 机制。
