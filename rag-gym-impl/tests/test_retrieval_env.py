import unittest
from unittest.mock import patch, MagicMock

from rag_gym_impl.upstream import import_symbols

Action = import_symbols("rag_gym.envs.action", "Action")[0]
from rag_gym_impl.retrieval_env import RetrievalEnv


@patch('rag_gym_impl.retrieval_env.RAGFlow')
class TestRetrievalEnv(unittest.TestCase):

    def test_step_with_search_action(self, MockRAGFlow: MagicMock):
        mock_client = MockRAGFlow.return_value
        mock_client.retrieve.return_value = [MagicMock(content="mocked doc chunk 1")]

        env = RetrievalEnv(dataset_ids=["test_id"])
        state, _ = env.reset(question="test question")
        search_action = Action(query="test query")

        next_state, terminated, truncated, _ = env.step(search_action)

        mock_client.retrieve.assert_called_once_with(
            question="test query",
            dataset_ids=["test_id"],
            top_k=5,
        )
        self.assertEqual(len(next_state.history), 1)
        self.assertEqual(next_state.history[0]["documents"], ["mocked doc chunk 1"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_step_with_finish_action(self, MockRAGFlow: MagicMock):
        env = RetrievalEnv(dataset_ids=["test_id"])
        state, _ = env.reset(question="test question")
        finish_action = Action(answer="final answer")

        next_state, terminated, truncated, _ = env.step(finish_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(next_state.answer, "final answer")

    def test_truncation_after_max_iter(self, MockRAGFlow: MagicMock):
        env = RetrievalEnv(dataset_ids=["test_id"], max_iter=1)
        state, _ = env.reset(question="test question")
        action = Action(query="test query")

        next_state, terminated, truncated, _ = env.step(action)

        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()


