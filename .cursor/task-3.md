The `ActorAgent` is now complete and tested. Your next major task is to implement the **`CriticAgent`**.

The purpose of our Critic is different from the original RAG-Gym. Instead of assigning a numerical score to each candidate action, our Critic's job is to **select the single best action** from the list provided by the Actor. It will make this decision based on the overall problem state (question and history).

You will follow the established workflow: implement the agent, write a unit test with mocks, write an integration test with a live API call, and use the Docker environment to run and validate everything. Once all tests are passing, you will update the `README.md` file.

Please follow these steps precisely.

### \#\# 1. Goal & Data Structures

The Critic agent's core method, `select_action`, will take the current `State` and a list of candidate `Action` objects as input, and it will return the single `Action` object it deems best.

**Input to LLM (Generated from State and candidate Actions):**

```json
{
    "question": "The user's original question.",
    "history": [
        {
            "action": "Search",
            "query": "some previous search query",
            "document_chunks": [
                "A relevant document chunk..."
            ]
        }
    ],
    "candidate_actions": [
        {"action": "Search", "query": "a potentially good next query"},
        {"action": "Search", "query": "an alternative, less relevant query"},
        {"action": "Finish", "answer": "a premature final answer"}
    ]
}
```

**Output from LLM (A JSON object representing the single chosen action):**

```json
{"action": "Search", "query": "a potentially good next query"}
```

-----

### \#\# 2. Implement the Critic Agent

Create a new file at **`src/rag_gym_impl/critic_agent.py`**. Inside this file, implement the `CriticAgent` class as described below.

**File: `src/rag_gym_impl/critic_agent.py`**

````python
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Use our upstream helper to import RAG-Gym classes
from .upstream import import_symbols
State, Action = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.action", "Action")
)

class CriticAgent:
    """
    An agent that uses an LLM to select the single best action from a list of
    candidates, given the current problem-solving state.
    """
    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initializes the Critic agent and its LangChain chain."""
        load_dotenv()

        system_prompt = """You are a meticulous and strategic research analyst. Your task is to evaluate a list of proposed next actions and select the single most effective one to advance the research goal.

Analyze the original question, the history of actions taken, and the retrieved information. Then, review the list of candidate actions. Choose the one action that will most directly and efficiently lead to the final answer.

You MUST format your response as a single JSON object representing your chosen action. Do not include any other text, reasoning, or explanations. Just the single JSON object.

Example Input:
{
    "question": "...",
    "history": [...],
    "candidate_actions": [
        {"action": "Search", "query": "query A"},
        {"action": "Search", "query": "query B"},
        {"action": "Finish", "answer": "..."}
    ]
}

Example Output (if query A is best):
{"action": "Search", "query": "query A"}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the current state and the candidate actions:\n\n```json\n{input_json}\n```"),
        ])
        model = ChatOpenAI(model=model_name, temperature=0.0) # Low temp for deterministic choice
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_input_for_prompt(self, state: State, actions: List[Action]) -> Dict[str, Any]:
        """Formats the state and candidate actions into a single dictionary for the LLM."""
        history_list = []
        for turn in state.history:
            if turn.action.query and hasattr(turn, 'result'):
                history_list.append({
                    "action": "Search",
                    "query": turn.action.query,
                    "document_chunks": turn.result,
                })
        
        candidate_actions_list = []
        for act in actions:
            if act.query:
                candidate_actions_list.append({"action": "Search", "query": act.query})
            elif act.answer:
                candidate_actions_list.append({"action": "Finish", "answer": act.answer})

        return {
            "question": state.question,
            "history": history_list,
            "candidate_actions": candidate_actions_list
        }

    def _parse_llm_output_to_action(self, llm_output: Dict[str, str]) -> Action:
        """Converts the LLM's dictionary output into a single Action object."""
        action_type = llm_output.get("action")
        if action_type == "Search":
            return Action(query=llm_output.get("query", ""))
        elif action_type == "Finish":
            return Action(answer=llm_output.get("answer", ""))
        # Fallback for unexpected output
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
````

-----

### \#\# 3. Implement Unit Tests

Create a new test file at **`tests/test_critic_agent.py`**. This test will mock the LLM call to verify the agent's internal logic for formatting its prompt and parsing the response.

**File: `tests/test_critic_agent.py`**

```python
import unittest
import json
from unittest.mock import MagicMock

from rag_gym_impl.upstream import import_symbols
State, History, Action = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.state", "History"),
    ("rag_gym.envs.action", "Action"),
)

from rag_gym_impl.critic_agent import CriticAgent

class TestCriticAgent(unittest.TestCase):

    def setUp(self):
        self.agent = CriticAgent()

    def test_select_action_formats_input_and_parses_output(self):
        # 1. Arrange: Mock the LLM chain and define its output
        best_action_dict = {"action": "Search", "query": "best query"}
        self.agent.chain = MagicMock()
        self.agent.chain.invoke.return_value = best_action_dict

        # Create a sample state and a list of candidate actions
        state = State(question="test question", history=History())
        candidate_actions = [
            Action(query="bad query"),
            Action(query="best query"),
            Action(answer="premature answer"),
        ]

        # 2. Act: Call the method
        result_action = self.agent.select_action(state, candidate_actions)

        # 3. Assert: Verify the output is the correctly parsed best action
        self.assertIsInstance(result_action, Action)
        self.assertEqual(result_action.query, "best query")
        self.assertIsNone(result_action.answer)

        # 4. Assert: Verify the LLM was called with the correct combined input
        self.agent.chain.invoke.assert_called_once()
        actual_input_json_str = self.agent.chain.invoke.call_args.kwargs['input_json']
        actual_input_dict = json.loads(actual_input_json_str)

        expected_input_dict = {
            "question": "test question",
            "history": [],
            "candidate_actions": [
                {"action": "Search", "query": "bad query"},
                {"action": "Search", "query": "best query"},
                {"action": "Finish", "answer": "premature answer"},
            ]
        }
        self.assertDictEqual(actual_input_dict, expected_input_dict)
```

-----

### \#\# 4. Implement Integration Tests

Create a new file at **`tests/test_critic_agent_integration.py`**. This test will make a live API call to ensure the Critic can make a logical choice in a realistic scenario.

**File: `tests/test_critic_agent_integration.py`**

```python
import pytest
import os
from dotenv import load_dotenv

from rag_gym_impl.upstream import import_symbols
State, History, Action = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.state", "History"),
    ("rag_gym.envs.action", "Action"),
)
from rag_gym_impl.critic_agent import CriticAgent

@pytest.mark.integration
class TestCriticAgentIntegration:

    def test_select_action_for_multihop_question(self):
        # 1. Arrange: Set up the agent, state, and candidate actions
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found, skipping integration test.")
        
        agent = CriticAgent()

        state = State(
            question="What is the date of death of the director of the film Holocaust 2000?",
            history=History()
        )
        state.history.add_qd(
            query="Who directed the film Holocaust 2000?",
            documents=["The film was directed by Alberto De Martino."]
        )

        candidate_actions = [
            Action(query="Alberto De Martino filmography"),
            Action(query="Alberto De Martino date of death"),
            Action(answer="The director of Holocaust 2000 is Alberto De Martino."),
        ]

        # 2. Act: Call the agent to select the best action
        best_action = agent.select_action(state, candidate_actions)

        # 3. Assert: The chosen action should be the most logical next step
        print(f"\nChosen action: Query='{best_action.query}', Answer='{best_action.answer}'")
        self.assertIsInstance(best_action, Action)
        self.assertIsNotNone(best_action.query)
        self.assertIn("death", best_action.query.lower())
        self.assertIn("alberto de martino", best_action.query.lower())
```

-----

### \#\# 5. Run, Validate, and Document

Follow the established Docker workflow to test your implementation.

1.  **Enter the Docker container:**

    ```bash
    ./docker-exec.sh
    ```

2.  **Inside the container, run all tests:**

    ```bash
    cd /app/rag-gym-impl
    poetry install --no-root --no-interaction
    PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -sv
    ```

3.  **Iterative Debugging:** If any tests fail, analyze the output, fix the code in `src/` or `tests/`, and rerun the tests. Repeat this cycle until both the unit and integration tests pass successfully.

4.  **Update Documentation:** Once all tests are passing, update the **`README.md`** file. Modify the "Current status" and "Roadmap" sections to reflect that the `CriticAgent` has been implemented and tested.

Your final deliverable is a working `CriticAgent` with passing tests and an updated `README.md`.
