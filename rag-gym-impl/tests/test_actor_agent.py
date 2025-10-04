import unittest
import json
from unittest.mock import MagicMock
import os

from rag_gym_impl.upstream import import_symbols

State, History = import_symbols("rag_gym.envs.state", "State", "History")
Action = import_symbols("rag_gym.envs.action", "Action")[0]

from rag_gym_impl.actor_agent import ActorAgent


class TestActorAgent(unittest.TestCase):

    def setUp(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        self.agent = ActorAgent()

    def test_generate_actions_formats_input_and_parses_output(self):
        # Arrange: mock LLM output and chain
        mock_llm_output = [
            {"action": "Search", "query": "what is the alpaca habitat"},
            {"action": "Finish", "answer": "Llamas are bigger."},
        ]
        self.agent.chain = MagicMock()
        self.agent.chain.invoke.return_value = mock_llm_output

        # Build a state with history
        question = "llama vs alpaca"
        history = History()
        history.add_qd(
            query="physical differences",
            documents=["Llamas are large.", "Alpacas have fine fiber."],
        )
        initial_state = State(question=question, history=history)

        # Act
        result_actions = self.agent.generate_actions(initial_state)

        # Assert: parsed actions
        self.assertIsInstance(result_actions, list)
        self.assertEqual(len(result_actions), 2)

        search_action = result_actions[0]
        self.assertIsInstance(search_action, Action)
        self.assertEqual(search_action.query, "what is the alpaca habitat")
        self.assertIsNone(search_action.answer)

        finish_action = result_actions[1]
        self.assertIsInstance(finish_action, Action)
        self.assertEqual(finish_action.answer, "Llamas are bigger.")
        self.assertIsNone(finish_action.query)

        # Assert: chain invoked with correctly formatted state_json
        self.agent.chain.invoke.assert_called_once()
        args, kwargs = self.agent.chain.invoke.call_args
        # input dict is first positional arg
        self.assertGreaterEqual(len(args), 1)
        actual_input_json_str = args[0]["state_json"]
        actual_input_dict = json.loads(actual_input_json_str)

        expected_input_dict = {
            "question": "llama vs alpaca",
            "history": [
                {
                    "action": "Search",
                    "query": "physical differences",
                    "document_chunks": [
                        "Llamas are large.",
                        "Alpacas have fine fiber.",
                    ],
                }
            ],
        }
        self.assertDictEqual(actual_input_dict, expected_input_dict)


if __name__ == "__main__":
    unittest.main()


