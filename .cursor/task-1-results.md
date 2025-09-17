**Cursor**

I'll scan the existing helper and upstream modules to align our implementation with how history and actions are represented before adding files and tests. Then I'll add dependencies, implement the actor, and write tests, running them afterward.

I'll open the project config and current package files to prepare for adding dependencies, the new actor implementation, and tests.

I'll quickly search the vendored upstream for any `Turn` class or different history structure to ensure our implementation and tests match the actual code.

I'll create a task list to track the steps, then add the dependencies, implement the actor agent and tests, and run the test suite.

I'm going to add the dependencies to `pyproject.toml`, implement `ActorAgent` at `src/rag_gym_impl/actor_agent.py`, create unit tests at `tests/test_actor_agent.py`, and update the root `README.md`.

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

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

```

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
            "[\n  {\"action\": \"Search\", \"query\": \"specific details about topic X\"},\n  {\"action\": \"Finish\", \"answer\": \"The final answer is...\"}\n]"
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

I'll add the new unit test file and update the README, then check for any linter issues in the new code.

```python
import unittest
import json
from unittest.mock import MagicMock

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

```markdown
# rag-gym-impl

A lightweight playground to prototype custom RAG-Gym-style agents using LLM calls. We vendor the original RAG-Gym repository as a git submodule and reuse multiple upstream components (e.g., state representation, actions, utilities). We intentionally do not rely on the upstream environment implementation, because we plan to integrate an external retriever stack (**RAGFlow**) and its datasets for information access.

## How this differs from the original RAG-Gym

- **Upstream reuse, selectively**: We load multiple modules directly from the upstream source (e.g., `rag_gym.envs.state`, `rag_gym.envs.action`, agent utilities, and more over time). We avoid importing the upstream top-level package to keep optional heavy dependencies (like `transformers`) out of the minimal example workflow when they aren’t required.
- **Custom environment layer**: In the original RAG-Gym, the environment is responsible for retrieval and observation generation. Here, we will not use the upstream environment. Instead, we will integrate with an external **RAGFlow** system and its datasets for retrieval and corpora access.
- **LLM-based agent loop**: Rather than using the upstream agent implementations, we will prototype agents that:
  - Generate action candidates from the current state using an LLM.
  - Mimic a critic by using an LLM to select a single action to promote and apply.
  - Continue the outer MDP loop until termination.

## Current status

- Minimal example that constructs a `State` using upstream code.
- Centralized import helper (`rag_gym_impl.upstream`) to load upstream modules/symbols without installing the upstream package.
- Docker + Poetry setup for reproducible runs.
- Actor agent implemented at `rag_gym_impl.actor_agent.ActorAgent` using LangChain + `gpt-5-nano` to generate candidate `Action`s.
- Unit tests for Actor agent at `rag_gym-impl/tests/test_actor_agent.py` (mocked LLM calls, no external API).

## Local development

Initialize git submodule (original RAG-Gym repository)

```bash
git submodule update --init --recursive
```

Build and run the container:

```bash
./docker-build.sh
./docker-run.sh
```

Run the example inside the container:

```bash
# from host
docker exec rag-gym bash -lc 'cd /app/rag-gym-impl && poetry install --no-root --no-interaction && poetry run python src/rag_gym_impl/main.py'
```

Expected output is the JSON representation of a simple `State`.

### Run tests

```bash
poetry install --no-root --no-interaction
poetry run pytest
```

## Importing upstream modules

To avoid importing the entire upstream `rag_gym` package (and transitively pulling optional dependencies), we use a small helper:

```python
from rag_gym_impl.upstream import import_symbol

State = import_symbol("rag_gym.envs.state", "State")
```

This loads `RAG-Gym/rag_gym/envs/state.py` directly from the submodule. As the project evolves, you can similarly import other upstream classes (e.g., actions, utilities) using `import_symbol` or `import_symbols` without changing `PYTHONPATH` or installing the upstream package.

## Roadmap

- Integrate RAGFlow for retrieval and datasets, replacing upstream env.
- Implement an LLM-driven critic to promote actions.
- Wire up the outer MDP loop with termination criteria and logging.
- Add tests and a small end-to-end example.

```

I'll add pytest as a dev dependency so we can run the tests, then install Poetry and project dependencies, and execute the test suite.

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

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

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

- I added deps to `pyproject.toml`, implemented `ActorAgent`, wrote tests, and updated `README.md`. Next I'll run the tests locally once Poetry is available.

What I did:
- Actor agent implemented at `src/rag_gym_impl/actor_agent.py` using LangChain `ChatOpenAI` and `JsonOutputParser`.
- Tests added at `tests/test_actor_agent.py` using a mocked chain; no live API calls.
- `pyproject.toml` updated with `langchain`, `langchain-openai`, `python-dotenv`, and dev `pytest`.
- `README.md` updated to reflect new components and test instructions.

To run tests on your machine:
- If Poetry isn’t installed, install it: `sudo apt install python3-poetry`
- Then install deps and run tests:
```bash
cd /home/laki/personal/rag-gym-impl/rag-gym-impl
poetry install --no-root --no-interaction
poetry run pytest
```

- All TODOs completed except running tests. Once you install Poetry and run pytest, let me know if anything fails and I’ll fix fast.

```shellscript
cd /home/laki/personal/rag-gym-impl/rag-gym-impl
poetry install --no-root --no-interaction
poetry run pytest
```

---

**User**

Why don't to use docker to run the tests? It is mentioned in the `@README.md` very clearly. I do not want to install Poetry or Python, I want to unitize Docker for development and also future in deployment of this project. Please use Docker.

```
$ ./docker-exec.sh 
root@dfe3116e461b:/app# cd rag-gym-impl/
root@dfe3116e461b:/app/rag-gym-impl# poetry install --no-root --no-interaction
Installing dependencies from lock file

No dependencies to install or update
root@dfe3116e461b:/app/rag-gym-impl# poetry run pytest
==================================================================================== test session starts =====================================================================================
platform linux -- Python 3.12.11, pytest-8.4.2, pluggy-1.6.0
rootdir: /app/rag-gym-impl
configfile: pyproject.toml
plugins: langsmith-0.4.28, anyio-4.10.0
collected 0 items / 1 error                                                                                                                                                                  

=========================================================================================== ERRORS ===========================================================================================
_________________________________________________________________________ ERROR collecting tests/test_actor_agent.py _________________________________________________________________________
ImportError while importing test module '/app/rag-gym-impl/tests/test_actor_agent.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_actor_agent.py:6: in <module>
    from rag_gym_impl.upstream import import_symbols
E   ModuleNotFoundError: No module named 'rag_gym_impl'
================================================================================== short test summary info ===================================================================================
ERROR tests/test_actor_agent.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
====================================================================================== 1 error in 0.07s ======================================================================================
```

Please utilize Docker.
There are errors when running pytest.
Resolve the issue, try to run again, iterate until you resolve all the issues and run the tests successfully.

---

**Cursor**

I'll use Docker to run the tests and debug the import path issue. I'll first inspect the docker scripts to understand how to correctly exec into the container and set PYTHONPATH, then run pytest with PYTHONPATH set.

I'll run the tests inside the Docker container, first updating the Poetry lock and installing deps, then executing pytest with PYTHONPATH pointing to `src`.

---

**User**

Anyway finally I can run the pytests with following commands:-
```
./docker-exec.sh

# Inside Docker container
poetry install --no-root --no-interaction
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest
```
