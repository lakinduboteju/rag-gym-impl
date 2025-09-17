All core components—Actor, Critic, and the Retrieval Environment—are now implemented and individually tested. Your final and most important task is to **bring everything together in a full end-to-end (E2E) integration test**. This test will simulate the entire agentic loop solving a multi-hop question.

Additionally, you will **update the `README.md` file** with detailed, step-by-step instructions for setting up the local development environment, including the RAGFlow instance, so that other developers can run these integration tests.

Please follow these two major steps precisely.

### \#\# Part 1: Update the README.md

First, modify the `README.md` file. Add a new, detailed section titled **"Running Integration Tests: Full Environment Setup"**. This section must guide a new user through the entire process of setting up the necessary external services.

Please insert the following markdown into `README.md`, likely after the "Local development" section.

````markdown
### Running Integration Tests: Full Environment Setup

The integration tests for the `CriticAgent`, `RetrievalEnv`, and the full E2E loop require a live OpenAI API key and a running instance of RAGFlow. Follow these steps to set up your environment:

**1. Set up the `.env` File**

At the root of the `rag-gym-impl` project, ensure you have a `.env` file containing your OpenAI API key:

```env
OPENAI_API_KEY="sk-..."
````

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
      * Add your OpenAI model details (e.g., `gpt-5-nano`) and provide your OpenAI API key.

  * **Generate RAGFlow API Key:**

      * Navigate to `https://localhost/user-setting/api`.
      * Generate a new API key.
      * Copy this key and add it to your `.env` file in the `rag-gym-impl` project:

    <!-- end list -->

    ```env
    OPENAI_API_KEY="sk-..."
    RAGFLOW_API_KEY="your-ragflow-api-key-here"
    ```

**4. Run the Tests**

With the setup complete, you can now run the integration tests from within this project's Docker container as described previously.

````

---

### ## Part 2: Implement the End-to-End Agent Loop Test

Now, create the final E2E test. This test will initialize all components and run the full agentic loop to answer a multi-hop question, using the live RAGFlow instance for retrieval.

Create a new file at **`tests/test_e2e_agent_loop.py`**.

**File: `tests/test_e2e_agent_loop.py`**
```python
import pytest
import os
import time
from dotenv import load_dotenv

from ragflow_sdk import RAGFlow
from rag_gym_impl.upstream import import_symbols
State = import_symbols("rag_gym.envs.state", "State")[0]

# Import our implemented agent and environment classes
from rag_gym_impl.actor_agent import ActorAgent
from rag_gym_impl.critic_agent import CriticAgent
from rag_gym_impl.retrieval_env import RetrievalEnv

@pytest.fixture(scope="module")
def e2e_ragflow_setup():
    """
    Sets up a temporary RAGFlow dataset with specific documents needed
    for the E2E multi-hop test. Cleans up the dataset afterward.
    """
    load_dotenv()
    api_key = os.getenv("RAGFLOW_API_KEY")
    base_url = "http://ragflow-server:9380"
    
    if not api_key or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("RAGFLOW_API_KEY or OPENAI_API_KEY not found, skipping E2E test.")
    
    rag_client = RAGFlow(api_key=api_key, base_url=base_url)
    
    dataset_name = f"e2e_test_dataset_{int(time.time())}"
    print(f"\nCreating E2E test dataset: {dataset_name}")
    dataset = rag_client.create_dataset(name=dataset_name)
    
    try:
        # Define documents that contain the necessary info, plus some noise
        docs_to_upload = [
            {"display_name": "doc1.txt", "blob": "Holocaust 2000 is a 1977 horror film directed by the Italian filmmaker Alberto De Martino.".encode('utf-8')},
            {"display_name": "doc2.txt", "blob": "Alberto De Martino (12 June 1929 – 31 October 2015) was a prolific director of genre films.".encode('utf-8')},
            {"display_name": "doc3_noise.txt", "blob": "The New York Times is a daily newspaper based in New York City.".encode('utf-8')},
        ]
        
        dataset.upload_documents(docs_to_upload)
        print("Waiting for RAGFlow to process documents...")
        time.sleep(10) # Allow time for parsing and indexing
        
        yield dataset.id
        
    finally:
        print(f"\nDeleting E2E test dataset: {dataset_name}")
        rag_client.delete_dataset(dataset_id=dataset.id)

@pytest.mark.integration
class TestEndToEndAgentLoop:

    def test_multi_hop_question_answering_loop(self, e2e_ragflow_setup):
        """
        Tests the full Actor-Critic-Environment loop on a multi-hop question.
        """
        # 1. Arrange: Initialize all components
        dataset_id = e2e_ragflow_setup
        
        actor = ActorAgent()
        critic = CriticAgent()
        env = RetrievalEnv(dataset_ids=[dataset_id], max_iter=3)
        
        question = "What is the date of death of the director of the film Holocaust 2000?"
        state, _ = env.reset(question=question)

        # 2. Act & Assert: Run the agentic loop
        for i in range(env.max_iter):
            print(f"\n----- LOOP {i+1} -----")
            
            # Actor generates actions
            candidate_actions = actor.generate_actions(state)
            print(f"Actor generated {len(candidate_actions)} actions.")
            
            # Critic selects the best action
            best_action = critic.select_action(state, candidate_actions)
            print(f"Critic selected action: Query='{best_action.query}', Answer='{best_action.answer}'")
            
            # Environment executes the action
            state, terminated, truncated, _ = env.step(best_action)

            # --- Assertions for each step ---
            if i == 0: # First loop should identify the director
                assert "alberto de martino" in state.history[0]["documents"][0].lower()
            
            if i == 1: # Second loop should find the date of death
                retrieved_docs_text = " ".join(state.history[1]["documents"]).lower()
                assert "2015" in retrieved_docs_text
            
            if terminated or truncated:
                print("Loop terminated or truncated.")
                break
        
        # 3. Final Assertion: Check the final answer
        print(f"Final Answer from state: {state.answer}")
        assert state.answer is not None, "Agent should have produced a final answer."
        final_answer = state.answer.lower()
        assert "alberto de martino" in final_answer
        assert "2015" in final_answer
````

-----

### \#\# Part 3: Final Validation

You have now added the final test and updated the documentation. Perform one last validation run.

1.  **Enter the Docker container:**
    ```bash
    ./docker-exec.sh
    ```
2.  **Inside the container, run all tests, ensuring the new E2E test passes:**
    ```bash
    cd /app/rag-gym-impl
    poetry install --no-root --no-interaction
    PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -sv
    ```
3.  **Iterative Debugging:** The E2E test is complex. If it fails, carefully examine the `print` statements from each loop iteration. The problem could be in the Actor's action generation, the Critic's choice, or the documents retrieved by the Environment. Debug and fix until the entire test suite passes.

Your final deliverable is the new E2E test file and the updated `README.md`, with all tests passing in the Docker environment.
