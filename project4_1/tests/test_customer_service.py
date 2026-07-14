import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.cli import run_turn
from src.db import initialize_database
from src.graph import build_app
from src.llm import ScriptedToolCallingModel
from src.rag import build_policy_retriever
from src.tools import build_tools


class CustomerServiceAgentTest(unittest.TestCase):
    def build_test_app(self, tmp_path: Path):
        db_path = tmp_path / "orders.db"
        checkpoint_path = tmp_path / "checkpoints.db"
        initialize_database(db_path)
        return build_app(
            model=ScriptedToolCallingModel(),
            tools=build_tools(db_path=db_path, retriever=build_policy_retriever()),
            checkpoint_path=checkpoint_path,
        )

    def test_cli_keeps_conversation_memory_by_thread_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.build_test_app(Path(directory))

            first_answer = run_turn(app, conversation_id="user_001", user_input="我要查订单")
            second_answer = run_turn(app, conversation_id="user_001", user_input="12345")

            self.assertIn("订单号", first_answer)
            self.assertIn("12345", second_answer)
            self.assertIn("shipped", second_answer)
            self.assertIn("Arrived at Beijing Sorting Center", second_answer)

    def test_graph_runs_check_order_tool_then_returns_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.build_test_app(Path(directory))

            result = app.invoke(
                {"messages": [HumanMessage(content="帮我查订单 67890")]},
                config={"configurable": {"thread_id": "order_tool_case"}},
            )

            messages = result["messages"]
            self.assertTrue(
                any(
                    isinstance(message, AIMessage)
                    and getattr(message, "tool_calls", None)
                    and message.tool_calls[0]["name"] == "check_order"
                    for message in messages
                )
            )
            self.assertTrue(
                any(isinstance(message, ToolMessage) and "67890" in message.content for message in messages)
            )
            self.assertIn("pending_payment", messages[-1].content)

    def test_policy_question_uses_rag_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.build_test_app(Path(directory))

            result = app.invoke(
                {"messages": [HumanMessage(content="不喜欢可以退货吗？")]},
                config={"configurable": {"thread_id": "policy_case"}},
            )

            messages = result["messages"]
            self.assertTrue(
                any(
                    isinstance(message, AIMessage)
                    and getattr(message, "tool_calls", None)
                    and message.tool_calls[0]["name"] == "search_policy"
                    for message in messages
                )
            )
            self.assertIn("7 天内", messages[-1].content)

    def test_unrelated_question_returns_fixed_boundary_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self.build_test_app(Path(directory))

            result = app.invoke(
                {"messages": [HumanMessage(content="讲个笑话")]},
                config={"configurable": {"thread_id": "fallback_case"}},
            )

            self.assertEqual(result["messages"][-1].content, "我可以帮您查询订单或解答政策相关问题。")


if __name__ == "__main__":
    unittest.main()
