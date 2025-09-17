import unittest
import json
from unittest.mock import MagicMock

from rag_gym_impl.upstream import import_symbols

State, History = import_symbols("rag_gym.envs.state", "State", "History")
Action = import_symbols("rag_gym.envs.action", "Action")[0]

from rag_gym_impl.critic_agent import CriticAgent


class TestCriticAgent(unittest.TestCase):

    def setUp(self):
        self.agent = CriticAgent()

    def test_select_action_formats_input_and_parses_output(self):
        # Arrange
        best_action_dict = {"action": "Search", "query": "best query"}
        self.agent.chain = MagicMock()
        self.agent.chain.invoke.return_value = best_action_dict

        state = State(question="test question", history=History())
        candidate_actions = [
            Action(query="bad query"),
            Action(query="best query"),
            Action(answer="premature answer"),
        ]

        # Act
        result_action = self.agent.select_action(state, candidate_actions)

        # Assert
        self.assertIsInstance(result_action, Action)
        self.assertEqual(result_action.query, "best query")
        self.assertIsNone(result_action.answer)

        self.agent.chain.invoke.assert_called_once()
        args, kwargs = self.agent.chain.invoke.call_args
        self.assertGreaterEqual(len(args), 1)
        actual_input_json_str = args[0]["input_json"]
        actual_input_dict = json.loads(actual_input_json_str)

        expected_input_dict = {
            "question": "test question",
            "history": [],
            "candidate_actions": [
                {"action": "Search", "query": "bad query"},
                {"action": "Search", "query": "best query"},
                {"action": "Finish", "answer": "premature answer"},
            ],
        }
        self.assertDictEqual(actual_input_dict, expected_input_dict)


if __name__ == "__main__":
    unittest.main()


