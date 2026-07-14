from langchain_core.messages import AIMessage, HumanMessage

from src.app_factory import create_app


def run_turn(app, conversation_id: str, user_input: str) -> str:
    """使用 thread_id 长记忆执行一轮 CLI 对话。"""
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": conversation_id}},
    )

    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return message.content
    return ""


def main() -> None:
    app = create_app()
    conversation_id = input("请输入对话ID: ").strip() or "default"
    print("输入 exit 或 quit 结束。")

    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("助手: 再见。")
            break
        if not user_input:
            continue

        answer = run_turn(app, conversation_id=conversation_id, user_input=user_input)
        print(f"助手: {answer}")


if __name__ == "__main__":
    main()
