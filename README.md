# astrbot_plugin_deepseek_search
让 astrbot llm 接入 deepseek 官方搜索工具

## 使用
1. 安装插件（放入 /AstrBot/data/plugins/ 并加载）。
2. 配置 API Key：
   - 方式 A：环境变量 `DEEPSEEK_API_KEY`
   - 方式 B：插件配置 `api_key`
3. 插件自动注册 LLM 工具 `deepseek_web_search(query)`，模型在需要实时信息时自动调用，返回搜索结果正文 + 引用来源。
4. 可在配置中调整 `model`（deepseek-chat / deepseek-v4-flash / deepseek-v4-pro）与 `base_url`。
