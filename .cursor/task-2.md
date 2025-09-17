The unit tests for the `ActorAgent` are passing, which confirms our internal logic is correct. Your next task is to write an **integration test** to validate the agent's real-world performance by making a live API call to the OpenAI `gpt-5-mini` model.

The goal is to verify that the agent can reason correctly in a **multi-hop question-answering scenario**. You will create a new test file, use the Docker environment for execution, and follow the iterative "run, analyze, fix" cycle until the test passes successfully.

Please follow these steps precisely.

### \#\# 1. Prerequisites: API Key

This is an integration test and requires a valid `OPENAI_API_KEY`. Ensure that the `.env` file at the project root (`/app/rag-gym-impl/.env`) contains a valid key. The test will fail without it.

### \#\# 2. Implement the Integration Test

Create a new test file at **`tests/test_actor_agent_integration.py`**. We are using a separate file to distinguish slow, network-dependent tests from fast unit tests.

Inside this file, implement the test as described below. The key is to set up a state that represents the halfway point in solving a multi-hop question and then assert that the LLM generates a logical next step.

**File: `tests/test_actor_agent_integration.py`**

```python
import pytest
import os

# Import necessary upstream classes for creating the test state
from rag_gym_impl.upstream import import_symbols
State, History, Action = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.state", "History"),
    ("rag_gym.envs.action", "Action"),
)

from rag_gym_impl.actor_agent import ActorAgent

# Mark this as an integration test to allow for selective runs
@pytest.mark.integration
class TestActorAgentIntegration:

    def test_generate_actions_for_multihop_question(self):
        """
        Tests the ActorAgent's ability to generate a logical next-step query
        by making a real API call to the OpenAI model.
        """
        # 1. Arrange: Instantiate the agent and create the multi-hop state
        # Ensure OPENAI_API_KEY is loaded from .env for this live test
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found, skipping integration test.")

        agent = ActorAgent()

        # This state mimics having already found the director's name (Alberto De Martino)
        # and now needing to find his date of death.
        question = "What is the date of death of the director of the film Holocaust 2000?"
        history = History()
        history.add_qd(
            query="Who directed the film Holocaust 2000?",
            documents=[
                "Holocaust 2000 is a 1977 Italian-British horror film directed by Alberto De Martino.",
                "The filmography of Alberto De Martino includes many popular Italian genre films of the 1960s and 1970s."
            ]
        )
        initial_state = State(question=question, history=history)

        # 2. Act: Call the agent to generate actions, triggering a real LLM call
        result_actions = agent.generate_actions(initial_state)

        # 3. Assert: Verify the generated actions are logical and well-formed
        self.assertIsInstance(result_actions, list)
        self.assertGreater(len(result_actions), 0, "Agent should generate at least one action.")

        # Check that at least one action is a search query
        search_actions = [action for action in result_actions if action.query]
        self.assertGreater(len(search_actions), 0, "Agent should produce at least one Search action.")

        # Check for semantic correctness of the generated query.
        # A good next step would be to search for the director's date of death.
        # We check for keywords to make the test robust against minor LLM output variations.
        query_text = " ".join([action.query.lower() for action in search_actions])
        
        print(f"\nGenerated search queries: {[action.query for action in search_actions]}")

        assert "alberto de martino" in query_text, "Query should mention the director's name."
        assert ("death" in query_text or "died" in query_text), "Query should be about the director's death."

```

### \#\# 3. Run and Validate in Docker

Now, use the established Docker workflow to run your new integration test.

1.  **Enter the Docker container:**

    ```bash
    ./docker-exec.sh
    ```

2.  **Inside the container, install dependencies and run *only* the integration test:**

    ```bash
    # Ensure dependencies are installed
    poetry install --no-root --no-interaction

    # Run the integration test using its marker
    PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -m integration -sv
    ```

    *(Note: `-sv` is added for verbose output so you can see the `print()` statements and test results clearly.)*

3.  **Iterative Debugging:**

      * **Run the test.**
      * **If it fails, analyze the output.** The printed "Generated search queries" will show you exactly what the LLM produced. The failure might be because the LLM's prompt needs refinement, or the test's assertions are too strict.
      * **Fix the code.** Modify `src/rag_gym_impl/actor_agent.py` or `tests/test_actor_agent_integration.py` as needed.
      * **Repeat the cycle.** Rerun the test command inside Docker until it passes consistently.

Your final deliverable is a successfully passing integration test that proves the `ActorAgent` can perform a logical reasoning step with a live LLM.
