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
- Actor agent implemented at `rag_gym_impl.actor_agent.ActorAgent` using LangChain + `gpt-5-mini` to generate candidate `Action`s.
- Unit tests for Actor agent at `rag_gym-impl/tests/test_actor_agent.py` (mocked LLM calls, no external API).
- Integration tests for Actor agent at `rag_gym-impl/tests/test_actor_agent_integration.py` (with external OpenAI API connection)
- Critic agent implemented at `rag_gym_impl.actor_agent.CriticAgent` using LangChain + `gpt-5-mini` to select the best from candidate `Action`s.
- Unit tests for Critic agent at `rag_gym-impl/tests/test_critic_agent.py` (mocked LLM calls, no external API).
- Integration tests for Critic agent at `rag_gym-impl/tests/test_critic_agent_integration.py` (with external OpenAI API connection)

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
docker exec rag-gym bash -lc 'cd /app/rag-gym-impl && poetry install --no-root --no-interaction && PYTHONPATH=/app/rag-gym-impl/src poetry run python src/rag_gym_impl/main.py'
```

Expected output is the JSON representation of a simple `State`.

### Run tests

```bash
./docker-exec.sh

# Inside Docker container
poetry install --no-root --no-interaction

# Run all tests
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest

# Run only integration tests
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -m integration -sv

# Run only 1 test
PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -sv tests/test_critic_agent_integration.py

# Debugging while running a test
PYTHONPATH=/app/rag-gym-impl/src \
poetry run python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
-m pytest -sv tests/test_retrieval_env_integration.py
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
- Wire up the outer MDP loop with termination criteria and logging.
- Add tests and a small end-to-end example.
