Your primary task is to implement the **"Actor" agent** for our custom RAG-Gym project. First, gain context by reading the project goal in `@docs/1-goal.md` and the project structure in `@README.md`.

The Actor's role is to generate candidate actions based on the current problem state. You will use **LangChain** to interface with the **gpt-5-nano** model and write comprehensive **unit tests** using Python's `unittest.mock` to ensure correctness without making live API calls.

Please follow these steps precisely.

### \#\# 1. Goal & Data Structures

The Actor agent's main method, `generate_actions`, will accept a `State` object and return a list of `Action` objects. You must use the existing `State` and `Action` classes from the vendored RAG-Gym submodule.

  * **State Class**: `@RAG-Gym/rag_gym/envs/state.py`
  * **Action Class**: `@RAG-Gym/rag_gym/envs/action.py` (The base `Action` class)

**Input to LLM (Generated from `State` object):**

```json
{
    "question": "The user's original question.",
    "history": [
        {
            "action": "Search",
            "query": "some previous search query",
            "document_chunks": [
                "A relevant document chunk...",
                "Another relevant document chunk..."
            ]
        }
    ]
}
```

**Output from LLM (A JSON string to be parsed):**

```json
[
  {"action": "Search", "query": "a new, useful search query"},
  {"action": "Search", "query": "an alternative search query"},
  {"action": "Finish", "answer": "A potential final answer based on current info."}
]
```

-----

### \#\# 2. Add Project Dependencies

Add the necessary Python libraries to `pyproject.toml` by running this command in your terminal:

```bash
poetry add langchain langchain-openai python-dotenv
```

-----

### \#\# 3. Implement the Actor Agent

Create a new file at **`src/rag_gym_impl/actor_agent.py`**. Implement the `ActorAgent` class as specified below. This version correctly uses the base `Action` class for both "Search" and "Finish" operations.

**File: `src/rag_gym_impl/actor_agent.py`**

````python
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# Use our upstream helper to import RAG-Gym classes
from .upstream import import_symbols
State, Action = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.action", "Action")
)

class ActorAgent:
    """
    An agent that uses an LLM to generate a set of candidate actions
    based on the current problem-solving state.
    """
    def __init__(self, model_name: str = "gpt-5-nano"):
        """Initializes the Actor agent and its LangChain chain."""
        load_dotenv()  # Loads OPENAI_API_KEY from .env file

        system_prompt = """You are an expert research assistant. Your goal is to help answer a complex question by proposing the next best actions.
Based on the user's question and the history of previous actions and retrieved documents, generate a list of 3 to 5 potential next actions.
The valid actions are:
1. `Search(query: str)`: To search for new information. The query should be specific and targeted to fill knowledge gaps.
2. `Finish(answer: str)`: To conclude the research and provide a final answer. Only use this if you are confident you have enough information.

You MUST format your response as a valid JSON list of objects. Do not include any other text, just the JSON.
Example:
[
  {"action": "Search", "query": "specific details about topic X"},
  {"action": "Finish", "answer": "The final answer is..."}
]"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the current state:\n\n```json\n{state_json}\n```"),
        ])
        model = ChatOpenAI(model=model_name, temperature=0.7)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_state_for_prompt(self, state: State) -> Dict[str, Any]:
        """Converts the State object into a JSON-serializable dictionary."""
        history_list = []
        for turn in state.history:
            # A search action in history will have a query and a result
            if turn.action.query and hasattr(turn, 'result'):
                history_list.append({
                    "action": "Search",
                    "query": turn.action.query,
                    "document_chunks": turn.result,
                })
        
        return {
            "question": state.question,
            "history": history_list,
        }

    def _parse_llm_output_to_actions(self, llm_output: List[Dict[str, str]]) -> List[Action]:
        """Converts the LLM's dictionary output into a list of Action objects."""
        actions = []
        for item in llm_output:
            action_type = item.get("action")
            if action_type == "Search":
                actions.append(Action(query=item.get("query", "")))
            elif action_type == "Finish":
                actions.append(Action(answer=item.get("answer", "")))
        return actions

    def generate_actions(self, state: State) -> List[Action]:
        """
        Generates candidate actions for a given state.

        Args:
            state: The current State object from the RAG-Gym environment.
        Returns:
            A list of candidate Action objects.
        """
        formatted_state = self._format_state_for_prompt(state)
        state_json_string = json.dumps(formatted_state, indent=2)
        
        llm_output = self.chain.invoke({"state_json": state_json_string})
        
        return self._parse_llm_output_to_actions(llm_output)
````

-----

### \#\# 4. Implement Unit Tests

Create a new test file at **`tests/test_actor_agent.py`**. This updated test verifies that the agent correctly produces instances of the base `Action` class with the appropriate `query` or `answer` attributes set.

**File: `tests/test_actor_agent.py`**

```python
import unittest
import json
from unittest.mock import patch, MagicMock

# Import necessary upstream classes for creating test state
from rag_gym_impl.upstream import import_symbols
State, Turn, Action = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.state", "Turn"),
    ("rag_gym.envs.action", "Action"),
)

from rag_gym_impl.actor_agent import ActorAgent

class TestActorAgent(unittest.TestCase):

    def setUp(self):
        """Set up a reusable ActorAgent instance for tests."""
        self.agent = ActorAgent()

    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_generate_actions_formats_input_and_parses_output(self, mock_invoke: MagicMock):
        """
        Verify the agent correctly formats the state for the LLM and
        parses the LLM's JSON output into proper Action objects.
        """
        # 1. Define a mock LLM response (list of dictionaries)
        mock_llm_output = [
            {"action": "Search", "query": "what is the alpaca habitat"},
            {"action": "Finish", "answer": "Llamas are bigger."}
        ]
        mock_invoke.return_value = mock_llm_output

        # 2. Create a sample input State object with a search history
        question = "llama vs alpaca"
        history = [
            Turn(
                action=Action(query="physical differences"),
                result=["Llamas are large.", "Alpacas have fine fiber."]
            )
        ]
        initial_state = State(question=question, history=history)

        # 3. Call the method we are testing
        result_actions = self.agent.generate_actions(initial_state)

        # 4. Assert the result is correctly parsed into Action objects
        self.assertIsInstance(result_actions, list)
        self.assertEqual(len(result_actions), 2)

        # Verify the Search action
        search_action = result_actions[0]
        self.assertIsInstance(search_action, Action)
        self.assertEqual(search_action.query, "what is the alpaca habitat")
        self.assertIsNone(search_action.answer)

        # Verify the Finish action
        finish_action = result_actions[1]
        self.assertIsInstance(finish_action, Action)
        self.assertEqual(finish_action.answer, "Llamas are bigger.")
        self.assertIsNone(finish_action.query)


        # 5. Assert that the LLM was called with the correctly formatted input
        mock_invoke.assert_called_once()
        call_args, _ = mock_invoke.call_args
        
        actual_input_json_str = call_args['input']['state_json']
        actual_input_dict = json.loads(actual_input_json_str)

        expected_input_dict = {
            "question": "llama vs alpaca",
            "history": [
                {
                    "action": "Search",
                    "query": "physical differences",
                    "document_chunks": ["Llamas are large.", "Alpacas have fine fiber."]
                }
            ]
        }
        
        self.assertDictEqual(actual_input_dict, expected_input_dict)
```

-----

### \#\# 5. Run and Validate

Finally, execute the tests to ensure everything is working correctly.

1.  Apply the code changes to the specified files.
2.  Run the tests from your terminal at the project root:
    ```bash
    poetry run pytest
    ```
3.  **Iterative Fixing:** If tests fail, analyze the traceback, fix the code, and rerun the tests. Repeat until all tests pass.

Your final deliverable is a working `ActorAgent` class with passing unit tests.
After that re-write the `@README.md` with current status of the project.
