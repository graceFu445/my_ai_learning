# LangGraph 订单客服 Demo

这个 Demo 使用 LangGraph 实现标准 tool-calling agent 循环，由 LLM 判断是否调用工具，LangGraph 负责执行工具、保存多轮对话状态，并把工具结果交回 LLM 生成客服回复。

## 分层结构

- `src/db.py`：初始化本地 SQLite 订单库，提供订单查询函数。
- `src/rag.py`：提供本地政策知识库检索器，用于退款、退货、物流规则、质保、支付、订单修改等咨询。
- `src/tools.py`：把订单查询和政策检索封装成 LangChain tools：`check_order`、`search_policy`。
- `src/llm.py`：配置千问 OpenAI 兼容模型，并提供无 API Key 时的本地脚本模型。
- `src/graph.py`：定义 LangGraph 节点和边。
- `src/cli.py`：命令行入口，只负责读取对话 ID、用户输入并调用 graph。

## LangGraph 节点和边

图结构：

```text
START -> agent -> route_tools
                  |-- no tool_calls --> END
                  |
                  |-- has tool_calls --> tools -> agent
```

- `agent`：调用绑定工具后的 LLM。LLM 根据系统提示词和历史消息决定直接回复，或生成 `tool_calls`。
- `tools`：使用 LangGraph `ToolNode` 执行 `check_order` 或 `search_policy`。
- `route_tools`：只判断最后一条 AI 消息是否有 `tool_calls`，有则进入 `tools`，否则结束。

ReAct 体现在 `agent -> tools -> agent` 循环中：

- Reasoning：LLM 在 `agent` 节点内部判断下一步。
- Acting：LLM 输出结构化 `tool_calls`。
- Observation：`ToolNode` 执行工具并返回 `ToolMessage`。
- Final Answer：工具结果回到 `agent` 后，LLM 生成最终客服回复。

## CLI 使用

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

运行：

```bash
python3 demo.py
```

启动后输入对话 ID：

```text
请输入对话ID: user_001
```

CLI 会把这个 ID 作为 LangGraph checkpoint 的 `thread_id`，同一个 ID 会保留长记忆，不同 ID 互相隔离。

## CLI 测试样例

长记忆：

```text
请输入对话ID: user_001

用户: 我要查订单
助手: 请提供订单号。

用户: 12345
助手: 查询结果：订单 12345（Wireless Headphones）：当前状态为 shipped。物流信息：Arrived at Beijing Sorting Center。
```

政策/RAG：

```text
用户: 不喜欢可以退货吗？
助手: 根据政策知识库：退款政策：自签收之日起 7 天内，且商品未拆封，可发起退款申请。
```

兜底：

```text
用户: 讲个笑话
助手: 我可以帮您查询订单或解答政策相关问题。
```

## 运行测试

```bash
python3 -m unittest tests/test_customer_service.py
```
