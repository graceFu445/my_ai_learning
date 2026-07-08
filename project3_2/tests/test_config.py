import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import AppConfig, load_dotenv


class ConfigTest(unittest.TestCase):
    def test_load_dotenv_reads_key_value_pairs_without_overwriting_existing_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DASHSCOPE_API_KEY=from-file",
                        "NEO4J_URI=bolt://localhost:7687",
                        "EMPTY_VALUE=",
                        "# 被忽略的注释",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "from-env"}, clear=False):
                load_dotenv(env_path)

                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "from-env")
                self.assertEqual(os.environ["NEO4J_URI"], "bolt://localhost:7687")
                self.assertEqual(os.environ["EMPTY_VALUE"], "")

    def test_app_config_reads_values_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "secret",
                "NEO4J_URI": "bolt://db:7687",
                "NEO4J_USERNAME": "user",
                "NEO4J_PASSWORD": "pass",
            },
            clear=False,
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.dashscope_api_key, "secret")
        self.assertEqual(config.neo4j_uri, "bolt://db:7687")
        self.assertEqual(config.neo4j_username, "user")
        self.assertEqual(config.neo4j_password, "pass")

    def test_qwen_clients_use_model_names_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "QWEN_EMBEDDING_MODEL": "embedding-from-env",
                "QWEN_LLM_MODEL": "llm-from-env",
            },
            clear=False,
        ):
            from src.embedding import QwenEmbeddingClient
            from src.generator import QwenTextGenerator
            from src.relation_extractor import QwenRelationExtractor

            embedding = QwenEmbeddingClient()
            generator = QwenTextGenerator()
            extractor = QwenRelationExtractor()

        self.assertEqual(embedding.model, "embedding-from-env")
        self.assertEqual(generator.model, "llm-from-env")
        self.assertEqual(extractor.model, "llm-from-env")


if __name__ == "__main__":
    unittest.main()
