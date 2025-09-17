**Cursor**

I'll add the integration test file, then run it inside Docker with the correct PYTHONPATH and marker. If it fails due to API issues or prompt tuning, I'll adjust and rerun until it passes.

```python
import pytest
import os

# Import necessary upstream classes for creating the test state
from rag_gym_impl.upstream import import_symbols
State, History, Action = import_symbols(
    "rag_gym.envs.state", "State", "History"
), import_symbols("rag_gym.envs.action", "Action")

State = State[0]
History = History[1] if isinstance(History, tuple) else History
Action = Action[0]

from rag_gym_impl.actor_agent import ActorAgent


@pytest.mark.integration
class TestActorAgentIntegration:

    def test_generate_actions_for_multihop_question(self):
        """
        Tests the ActorAgent's ability to generate a logical next-step query
        by making a real API call to the OpenAI model.
        """
        # Ensure OPENAI_API_KEY is present for live test
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found, skipping integration test.")

        agent = ActorAgent()

        question = "What is the date of death of the director of the film Holocaust 2000?"
        history = History()
        history.add_qd(
            query="Who directed the film Holocaust 2000?",
            documents=[
                "Holocaust 2000 is a 1977 Italian-British horror film directed by Alberto De Martino.",
                "The filmography of Alberto De Martino includes many popular Italian genre films of the 1960s and 1970s.",
            ],
        )
        initial_state = State(question=question, history=history)

        result_actions = agent.generate_actions(initial_state)

        assert isinstance(result_actions, list)
        assert len(result_actions) > 0, "Agent should generate at least one action."

        search_actions = [action for action in result_actions if getattr(action, "query", None)]
        assert len(search_actions) > 0, "Agent should produce at least one Search action."

        query_text = " ".join([action.query.lower() for action in search_actions if action.query])

        print(f"\nGenerated search queries: {[action.query for action in search_actions]}")

        assert "alberto de martino" in query_text, "Query should mention the director's name."
        assert ("death" in query_text or "died" in query_text), "Query should be about the director's death."


```

I'll fix the integration test imports and ensure it loads the .env file before checking the API key. Then I'll run the integration test inside Docker with the proper PYTHONPATH and marker.

```python
import pytest
import os
from dotenv import load_dotenv

# Import necessary upstream classes for creating the test state
from rag_gym_impl.upstream import import_symbols
State, History = import_symbols("rag_gym.envs.state", "State", "History")
Action = import_symbols("rag_gym.envs.action", "Action")[0]

from rag_gym_impl.actor_agent import ActorAgent


@pytest.mark.integration
class TestActorAgentIntegration:

    def test_generate_actions_for_multihop_question(self):
        """
        Tests the ActorAgent's ability to generate a logical next-step query
        by making a real API call to the OpenAI model.
        """
        # Load .env to ensure OPENAI_API_KEY is present for live test
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found, skipping integration test.")

        agent = ActorAgent()

        question = "What is the date of death of the director of the film Holocaust 2000?"
        history = History()
        history.add_qd(
            query="Who directed the film Holocaust 2000?",
            documents=[
                "Holocaust 2000 is a 1977 Italian-British horror film directed by Alberto De Martino.",
                "The filmography of Alberto De Martino includes many popular Italian genre films of the 1960s and 1970s.",
            ],
        )
        initial_state = State(question=question, history=history)

        result_actions = agent.generate_actions(initial_state)

        assert isinstance(result_actions, list)
        assert len(result_actions) > 0, "Agent should generate at least one action."

        search_actions = [action for action in result_actions if getattr(action, "query", None)]
        assert len(search_actions) > 0, "Agent should produce at least one Search action."

        query_text = " ".join([action.query.lower() for action in search_actions if action.query])

        print(f"\nGenerated search queries: {[action.query for action in search_actions]}")

        assert "alberto de martino" in query_text, "Query should mention the director's name."
        assert ("death" in query_text or "died" in query_text), "Query should be about the director's death."



```

I'll fix the failing prompt formatting by escaping JSON braces in the system prompt, register the pytest marker in `pyproject.toml`, and then rerun the integration test inside Docker.

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


class ActorAgent:
    """
    An agent that uses an LLM to generate a set of candidate actions
    based on the current problem-solving state.
    """

    def __init__(self, model_name: str = "gpt-5-nano"):
        """Initializes the Actor agent and its LangChain chain."""
        load_dotenv()

        system_prompt = (
            "You are an expert research assistant. Your goal is to help answer a complex question by proposing the next best actions.\n"
            "Based on the user's question and the history of previous actions and retrieved documents, generate a list of 3 to 5 potential next actions.\n"
            "The valid actions are:\n"
            "1. `Search(query: str)`: To search for new information. The query should be specific and targeted to fill knowledge gaps.\n"
            "2. `Finish(answer: str)`: To conclude the research and provide a final answer. Only use this if you are confident you have enough information.\n\n"
            "You MUST format your response as a valid JSON list of objects. Do not include any other text, just the JSON.\n"
            "Example:\n"
            "[\n  {{\"action\": \"Search\", \"query\": \"specific details about topic X\"}},\n  {{\"action\": \"Finish\", \"answer\": \"The final answer is...\"}}\n]"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the current state:\n\n```json\n{state_json}\n```"),
            ]
        )
        model = ChatOpenAI(model=model_name, temperature=0.7)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_state_for_prompt(self, state: State) -> Dict[str, Any]:
        """Converts the State object into a JSON-serializable dictionary."""
        history_list: List[Dict[str, Any]] = []

        # Upstream History is iterable over dicts with keys query/documents
        for item in state.history:
            # Expecting dicts like {"query": str, "documents": list[...]} per vendored State
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

        return {
            "question": state.question,
            "history": history_list,
        }

    def _parse_llm_output_to_actions(self, llm_output: List[Dict[str, str]]) -> List[Action]:
        """Converts the LLM's dictionary output into a list of Action objects."""
        actions: List[Action] = []
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



```

```plaintext
[project]
name = "rag-gym-impl"
version = "0.1.0"
description = ""
authors = [
    {name = "Your Name",email = "you@example.com"}
]
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "python-dotenv>=1.0.0",
]

[tool.poetry]
packages = [
    {include = "rag_gym_impl", from = "src"},
]

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0.0"

[tool.pytest.ini_options]
markers = [
    "integration: marks tests that call external services (deselect with '-m "not integration"')",
]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

```

I'll open the TOML config to fix the markers array syntax issue, then rerun tests inside Docker.

```plaintext
[project]
name = "rag-gym-impl"
version = "0.1.0"
description = ""
authors = [
    {name = "Your Name",email = "you@example.com"}
]
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "python-dotenv>=1.0.0",
]

[tool.poetry]
packages = [
    {include = "rag_gym_impl", from = "src"},
]

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0.0"

[tool.pytest.ini_options]
markers = [
    'integration: marks tests that call external services (deselect with "-m "not integration"")',
]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

```

- Integration test added and passing in Docker.
- Fixed LangChain prompt variable issue by escaping JSON braces in the system prompt.
- Registered pytest marker to avoid warnings.
- Ran inside container with PYTHONPATH set: one integration test passed.

You can re-run:
```bash
./docker-exec.sh
cd /app/rag-gym-impl
poetry install --no-root --no-interaction
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -m integration -sv
```

```shellscript
./docker-exec.sh
cd /app/rag-gym-impl
poetry install --no-root --no-interaction
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -m integration -sv
```
