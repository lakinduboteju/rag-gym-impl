# rag-gym-impl

A lightweight playground to prototype custom RAG-Gym-style agents using LLM calls. We vendor the original RAG-Gym repository as a git submodule and reuse multiple upstream components (e.g., state representation, actions, utilities). We intentionally do not rely on the upstream environment implementation, because we plan to integrate an external retriever stack (**RAGFlow**) and its datasets for information access.

## How this differs from the original RAG-Gym

- **Upstream reuse, selectively**: We load multiple modules directly from the upstream source (e.g., `rag_gym.envs.state`, `rag_gym.envs.action`, agent utilities, and more over time). We avoid importing the upstream top-level package to keep optional heavy dependencies (like `transformers`) out of the minimal example workflow when they aren’t required.
- **Custom environment layer**: In the original RAG-Gym, the environment is responsible for retrieval and observation generation. Here, we replace the upstream environment with a custom `RetrievalEnv` that connects to an external **RAGFlow** service for retrieval and corpora access.
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
- Integration tests for Actor agent at `rag_gym-impl/tests/test_actor_agent_integration.py` (connecting to OpenAI API).

- Critic agent implemented at `rag_gym_impl.critic_agent.CriticAgent` selecting the best action from candidates.
- Unit tests for Critic agent at `rag_gym-impl/tests/test_critic_agent.py` (mocked LLM calls, no external API).
- Integration tests for Critic agent at `rag_gym-impl/tests/test_critic_agent_integration.py` (connecting to OpenAI API).

- Custom retrieval environment implemented at `rag_gym_impl.retrieval_env.RetrievalEnv` integrating with `ragflow-sdk`.
- Unit tests for `RetrievalEnv` at `rag_gym-impl/tests/test_retrieval_env.py` (mocked calls, no external API).
- Integration tests for `RetrievalEnv` at `rag_gym-impl/tests/test_retrieval_env_integration.py` (connecting to RAGFlow API).

- Full end-to-end integration test connecting the 2 Agents + `RetrievalEnv` at `rag_gym-impl/tests/test_e2e_agent_loop.py`.

---

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
cd /app/rag-gym-impl/
poetry install --no-root --no-interaction

# Run all tests
PYTHONPATH=$(pwd)/src poetry run pytest -m "not integration" -sv

# Run only integration tests
PYTHONPATH=$(pwd)/src poetry run pytest -m integration -sv

# Run only 1 test
PYTHONPATH=$(pwd)/src poetry run pytest -sv tests/test_critic_agent_integration.py

# Debugging while running a test
PYTHONPATH=$(pwd)/src \
poetry run python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
-m pytest -sv tests/test_retrieval_env_integration.py
# Use VS Code to connect debug client to the debugpy server running on Docker container
# Use .vscode/launch.json
```

### Running Integration Tests: Full Setup

The integration tests for the Agents and `RetrievalEnv`, and the full E2E loop require a live OpenAI API key and a running instance of RAGFlow. Follow these steps to set up your environment:

**1. Set up the `.env` File**

At the root of the `rag-gym-impl` project, ensure you have a `.env` file containing your OpenAI API key:

```env
OPENAI_API_KEY="sk-..."
```

**2. Clone and Run RAGFlow**

We use a specific version of RAGFlow. Clone it and check out the correct branch:

```bash
# Navigate to a directory outside of this project
git clone [https://github.com/lakinduboteju/ragflow.git](https://github.com/lakinduboteju/ragflow.git)
cd ragflow/

# Checkout the required branch
git checkout lakindu
```

Now, bring up the RAGFlow services using Docker Compose. This command must be run from the `ragflow/docker/` directory:

```bash
cd docker/
docker compose -f docker-compose-https.yml up -d
```

*(Note: Ensure this project's Docker container is running on the same `docker_ragflow` network to allow communication.)*

**3. Configure RAGFlow via UI**

Once RAGFlow is running, you must perform two manual configuration steps in your web browser:

  * **Add OpenAI Model:**

      * Navigate to `https://localhost/user-setting/model` (you may need to accept a self-signed certificate warning).
      * Add your OpenAI model details (e.g., `gpt-5-nano`, `text-embedding-3-small`) and provide your OpenAI API key.

  * **Generate RAGFlow API Key:**

      * Navigate to `https://localhost/user-setting/api`.
      * Generate a new API key.
      * Copy this key and add it to your `.env` file in the `rag-gym-impl` project:

    ```env
    OPENAI_API_KEY="sk-..."
    RAGFLOW_API_KEY="your-ragflow-api-key-here"
    ```

**4. Run the Tests**

With the setup complete, you can now run the integration tests from within this project's Docker container as described previously.

---

## Importing upstream modules

To avoid importing the entire upstream `rag_gym` package (and transitively pulling optional dependencies), we use a small helper:

```python
from rag_gym_impl.upstream import import_symbol

State = import_symbol("rag_gym.envs.state", "State")
```

This loads `RAG-Gym/rag_gym/envs/state.py` directly from the submodule. As the project evolves, you can similarly import other upstream classes (e.g., actions, utilities) using `import_symbol` or `import_symbols` without changing `PYTHONPATH` or installing the upstream package.

## Roadmap

- Integrate RAGFlow for retrieval and datasets, replacing upstream env.
- Expand RetrievalEnv capabilities (fallback retrievers, reranking, tracing).
- Wire up the outer MDP loop with termination criteria and logging.
- Add tests and a small end-to-end example.
