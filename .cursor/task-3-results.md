**Cursor**

I'll create the CriticAgent implementation and tests, then run the full test suite inside Docker with PYTHONPATH configured. If anything fails, I'll adjust and rerun until green.

```python
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]
Action = import_symbols("rag_gym.envs.action", "Action")[0]


class CriticAgent:
    """
    An agent that uses an LLM to select the single best action from a list of
    candidates, given the current problem-solving state.
    """

    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initializes the Critic agent and its LangChain chain."""
        load_dotenv()

        system_prompt = (
            "You are a meticulous and strategic research analyst. Your task is to evaluate a list of proposed next actions and select the single most effective one to advance the research goal.\n\n"
            "Analyze the original question, the history of actions taken, and the retrieved information. Then, review the list of candidate actions. Choose the one action that will most directly and efficiently lead to the final answer.\n\n"
            "You MUST format your response as a single JSON object representing your chosen action. Do not include any other text, reasoning, or explanations. Just the single JSON object.\n\n"
            "Example Input:\n"
            "{\n    \"question\": \"...\",\n    \"history\": [...],\n    \"candidate_actions\": [\n        {\"action\": \"Search\", \"query\": \"query A\"},\n        {\"action\": \"Search\", \"query\": \"query B\"},\n        {\"action\": \"Finish\", \"answer\": \"...\"}\n    ]\n}\n\n"
            "Example Output (if query A is best):\n"
            "{\"action\": \"Search\", \"query\": \"query A\"}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the current state and the candidate actions:\n\n```json\n{input_json}\n```"),
            ]
        )
        model = ChatOpenAI(model=model_name, temperature=0.0)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_input_for_prompt(self, state: State, actions: List[Action]) -> Dict[str, Any]:
        """Formats the state and candidate actions into a single dictionary for the LLM."""
        history_list: List[Dict[str, Any]] = []
        for item in state.history:
            query = item.get("query") if isinstance(item, dict) else None
            documents = item.get("documents") if isinstance(item, dict) else None
            if query is not None and documents is not None:
                history_list.append(
                    {
                        "action": "Search",
                        "query": query,
                        "document_chunks": documents,
                    }
                )

        candidate_actions_list: List[Dict[str, str]] = []
        for act in actions:
            if getattr(act, "query", None):
                candidate_actions_list.append({"action": "Search", "query": act.query})
            elif getattr(act, "answer", None):
                candidate_actions_list.append({"action": "Finish", "answer": act.answer})

        return {
            "question": state.question,
            "history": history_list,
            "candidate_actions": candidate_actions_list,
        }

    def _parse_llm_output_to_action(self, llm_output: Dict[str, str]) -> Action:
        """Converts the LLM's dictionary output into a single Action object."""
        action_type = llm_output.get("action")
        if action_type == "Search":
            return Action(query=llm_output.get("query", ""))
        if action_type == "Finish":
            return Action(answer=llm_output.get("answer", ""))
        return Action()

    def select_action(self, state: State, actions: List[Action]) -> Action:
        """
        Selects the best action from a list of candidates.

        Args:
            state: The current State object.
            actions: A list of candidate Action objects from the Actor.
        Returns:
            The single best Action object as chosen by the LLM.
        """
        formatted_input = self._format_input_for_prompt(state, actions)
        input_json_string = json.dumps(formatted_input, indent=2)

        llm_output = self.chain.invoke({"input_json": input_json_string})

        return self._parse_llm_output_to_action(llm_output)


```

```python
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
        actual_input_json_str = self.agent.chain.invoke.call_args.kwargs["input_json"]
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


```

```python
import pytest
import os
from dotenv import load_dotenv

from rag_gym_impl.upstream import import_symbols

State, History = import_symbols("rag_gym.envs.state", "State", "History")
Action = import_symbols("rag_gym.envs.action", "Action")[0]

from rag_gym_impl.critic_agent import CriticAgent


@pytest.mark.integration
class TestCriticAgentIntegration:

    def test_select_action_for_multihop_question(self):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found, skipping integration test.")

        agent = CriticAgent()

        state = State(
            question="What is the date of death of the director of the film Holocaust 2000?",
            history=History(),
        )
        state.history.add_qd(
            query="Who directed the film Holocaust 2000?",
            documents=["The film was directed by Alberto De Martino."],
        )

        candidate_actions = [
            Action(query="Alberto De Martino filmography"),
            Action(query="Alberto De Martino date of death"),
            Action(answer="The director of Holocaust 2000 is Alberto De Martino."),
        ]

        best_action = agent.select_action(state, candidate_actions)

        print(f"\nChosen action: Query='{best_action.query}', Answer='{best_action.answer}'")
        assert isinstance(best_action, Action)
        assert getattr(best_action, "query", None)
        assert "death" in best_action.query.lower()
        assert "alberto de martino" in best_action.query.lower()


```

I'll fix the Critic prompt escaping, update the unit test to inspect positional args, and remove the environment override in the Actor unit test to avoid polluting integration tests. Then I'll rerun the full test suite in Docker.

```python
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]
Action = import_symbols("rag_gym.envs.action", "Action")[0]


class CriticAgent:
    """
    An agent that uses an LLM to select the single best action from a list of
    candidates, given the current problem-solving state.
    """

    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initializes the Critic agent and its LangChain chain."""
        load_dotenv()

        system_prompt = (
            "You are a meticulous and strategic research analyst. Your task is to evaluate a list of proposed next actions and select the single most effective one to advance the research goal.\n\n"
            "Analyze the original question, the history of actions taken, and the retrieved information. Then, review the list of candidate actions. Choose the one action that will most directly and efficiently lead to the final answer.\n\n"
            "You MUST format your response as a single JSON object representing your chosen action. Do not include any other text, reasoning, or explanations. Just the single JSON object.\n\n"
            "Example Input:\n"
            "[[\n    \"question\": \"...\",\n    \"history\": [...],\n    \"candidate_actions\": [\n        {\"action\": \"Search\", \"query\": \"query A\"},\n        {\"action\": \"Search\", \"query\": \"query B\"},\n        {\"action\": \"Finish\", \"answer\": \"...\"}\n    ]\n]]\n\n"
            "Example Output (if query A is best):\n"
            "{{\"action\": \"Search\", \"query\": \"query A\"}}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the current state and the candidate actions:\n\n```json\n{input_json}\n```"),
            ]
        )
        model = ChatOpenAI(model=model_name, temperature=0.0)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_input_for_prompt(self, state: State, actions: List[Action]) -> Dict[str, Any]:
        """Formats the state and candidate actions into a single dictionary for the LLM."""
        history_list: List[Dict[str, Any]] = []
        for item in state.history:
            query = item.get("query") if isinstance(item, dict) else None
            documents = item.get("documents") if isinstance(item, dict) else None
            if query is not None and documents is not None:
                history_list.append(
                    {
                        "action": "Search",
                        "query": query,
                        "document_chunks": documents,
                    }
                )

        candidate_actions_list: List[Dict[str, str]] = []
        for act in actions:
            if getattr(act, "query", None):
                candidate_actions_list.append({"action": "Search", "query": act.query})
            elif getattr(act, "answer", None):
                candidate_actions_list.append({"action": "Finish", "answer": act.answer})

        return {
            "question": state.question,
            "history": history_list,
            "candidate_actions": candidate_actions_list,
        }

    def _parse_llm_output_to_action(self, llm_output: Dict[str, str]) -> Action:
        """Converts the LLM's dictionary output into a single Action object."""
        action_type = llm_output.get("action")
        if action_type == "Search":
            return Action(query=llm_output.get("query", ""))
        if action_type == "Finish":
            return Action(answer=llm_output.get("answer", ""))
        return Action()

    def select_action(self, state: State, actions: List[Action]) -> Action:
        """
        Selects the best action from a list of candidates.

        Args:
            state: The current State object.
            actions: A list of candidate Action objects from the Actor.
        Returns:
            The single best Action object as chosen by the LLM.
        """
        formatted_input = self._format_input_for_prompt(state, actions)
        input_json_string = json.dumps(formatted_input, indent=2)

        llm_output = self.chain.invoke({"input_json": input_json_string})

        return self._parse_llm_output_to_action(llm_output)



```

```python
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



```

```python
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
        self.agent = ActorAgent()

    def test_generate_actions_formats_input_and_parses_output(self):
        # Arrange: mock LLM output and chain
        mock_llm_output = [
            {"action": "Search", "query": "what is the alpaca habitat"},
            {"action": "Finish", "answer": "Llamas are bigger."},
        ]
        self.agent.chain = MagicMock()
        self.agent.chain.invoke.return_value = mock_llm_output

        # Build input state with history
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



```

I updated the unit and integration tests and made fixes to the critic prompt, and now I'll adjust the critic prompt to correctly escape JSON braces to avoid LangChain variable parsing issues, then rerun the Docker tests.

```python
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]
Action = import_symbols("rag_gym.envs.action", "Action")[0]


class CriticAgent:
    """
    An agent that uses an LLM to select the single best action from a list of
    candidates, given the current problem-solving state.
    """

    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initializes the Critic agent and its LangChain chain."""
        load_dotenv()

        system_prompt = (
            "You are a meticulous and strategic research analyst. Your task is to evaluate a list of proposed next actions and select the single most effective one to advance the research goal.\n\n"
            "Analyze the original question, the history of actions taken, and the retrieved information. Then, review the list of candidate actions. Choose the one action that will most directly and efficiently lead to the final answer.\n\n"
            "You MUST format your response as a single JSON object representing your chosen action. Do not include any other text, reasoning, or explanations. Just the single JSON object.\n\n"
            "Example Input:\n"
            "{{\n    \\\"question\\\": \\\"...\\\",\n    \\\"history\\\": [...],\n    \\\"candidate_actions\\\": [\n        {{\\\"action\\\": \\\"Search\\\", \\\"query\\\": \\\"query A\\\"}},\n        {{\\\"action\\\": \\\"Search\\\", \\\"query\\\": \\\"query B\\\"}},\n        {{\\\"action\\\": \\\"Finish\\\", \\\"answer\\\": \\\"...\\\"}}\n    ]\n}}\n\n"
            "Example Output (if query A is best):\n"
            "{{\\\"action\\\": \\\"Search\\\", \\\"query\\\": \\\"query A\\\"}}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the current state and the candidate actions:\n\n```json\n{input_json}\n```"),
            ]
        )
        model = ChatOpenAI(model=model_name, temperature=0.0)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_input_for_prompt(self, state: State, actions: List[Action]) -> Dict[str, Any]:
        """Formats the state and candidate actions into a single dictionary for the LLM."""
        history_list: List[Dict[str, Any]] = []
        for item in state.history:
            query = item.get("query") if isinstance(item, dict) else None
            documents = item.get("documents") if isinstance(item, dict) else None
            if query is not None and documents is not None:
                history_list.append(
                    {
                        "action": "Search",
                        "query": query,
                        "document_chunks": documents,
                    }
                )

        candidate_actions_list: List[Dict[str, str]] = []
        for act in actions:
            if getattr(act, "query", None):
                candidate_actions_list.append({"action": "Search", "query": act.query})
            elif getattr(act, "answer", None):
                candidate_actions_list.append({"action": "Finish", "answer": act.answer})

        return {
            "question": state.question,
            "history": history_list,
            "candidate_actions": candidate_actions_list,
        }

    def _parse_llm_output_to_action(self, llm_output: Dict[str, str]) -> Action:
        """Converts the LLM's dictionary output into a single Action object."""
        action_type = llm_output.get("action")
        if action_type == "Search":
            return Action(query=llm_output.get("query", ""))
        if action_type == "Finish":
            return Action(answer=llm_output.get("answer", ""))
        return Action()

    def select_action(self, state: State, actions: List[Action]) -> Action:
        """
        Selects the best action from a list of candidates.

        Args:
            state: The current State object.
            actions: A list of candidate Action objects from the Actor.
        Returns:
            The single best Action object as chosen by the LLM.
        """
        formatted_input = self._format_input_for_prompt(state, actions)
        input_json_string = json.dumps(formatted_input, indent=2)

        llm_output = self.chain.invoke({"input_json": input_json_string})

        return self._parse_llm_output_to_action(llm_output)



```

- All tests now pass in Docker, including the new Critic integration test.
- Fixes included: escaping JSON braces in `CriticAgent` prompt; adjusting unit test arg inspection; avoiding test key pollution.

To run everything:
```bash
./docker-exec.sh
cd /app/rag-gym-impl
poetry install --no-root --no-interaction
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -sv
```

Changes:
- Added `src/rag_gym_impl/critic_agent.py`
- Added tests: `tests/test_critic_agent.py`, `tests/test_critic_agent_integration.py`
- Minor fixes to prompts and tests to satisfy LangChain input variable validation

Would you like me to update `README.md` to reflect the CriticAgent implementation and how to run integration tests separately?

```shellscript
./docker-exec.sh
cd /app/rag-gym-impl
poetry install --no-root --no-interaction
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -sv
```
