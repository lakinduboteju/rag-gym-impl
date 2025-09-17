The Actor and Critic agents are complete. The final core component is the **Retrieval Environment**. Your task is to implement a new `RetrievalEnv` class that manages the agent's state and interacts with our live RAGFlow instance for document retrieval.

You will implement the class, write **both a unit test (using mocks) and an integration test (with a live connection)**, and validate the entire workflow using our Docker setup. Once all tests are passing, you will update the `README.md` file.

Please follow these steps precisely.

### \#\# 1. Design the Custom `RetrievalEnv`

Based on the RAG-Gym `Env` reference and our project's needs, here is the proposed design for our custom environment. We will simplify it to focus purely on state transitions and RAGFlow retrieval.

  * **Constructor (`__init__`)**: We will drop irrelevant parameters like `retriever_name` and `corpus_name`. Instead, we'll accept parameters needed for the `ragflow-sdk` and the retrieval process.
  * **`step` Method**: This is the core of the environment.
      * If it receives a `Search` action, it will call the RAGFlow API to retrieve documents and update the state's history.
      * If it receives a `Finish` action, it will simply mark the episode as terminated.
      * We will **drop the `reward` output** from the `step` method, as our agent loop does not use an environment-generated reward signal. The return signature will be `(next_state, terminated, truncated, info)`.

### \#\# 2. Add Project Dependencies

Add the RAGFlow Python SDK to the project.

```bash
poetry add ragflow-sdk
```

### \#\# 3. Implement the Retrieval Environment

Create a new file at **`src/rag_gym_impl/retrieval_env.py`**. Inside this file, implement the `RetrievalEnv` class as described below.

**File: `src/rag_gym_impl/retrieval_env.py`**

```python
import os
from typing import Tuple, Dict, Any, List

from ragflow_sdk import RAGFlow

from .upstream import import_symbols
State, Action, History = import_symbols(
    ("rag_gym.envs.state", "State"),
    ("rag_gym.envs.action", "Action"),
    ("rag_gym.envs.state", "History"),
)

class RetrievalEnv:
    """
    A custom environment that manages state and retrieves documents from a RAGFlow instance.
    """
    def __init__(
        self,
        dataset_ids: List[str],
        ragflow_base_url: str = "http://ragflow-server:9380",
        max_iter: int = 5,
        top_k: int = 5,
    ):
        """
        Initializes the environment and the RAGFlow client.

        Args:
            dataset_ids: A list of RAGFlow dataset IDs to retrieve from.
            ragflow_base_url: The base URL for the RAGFlow API.
            max_iter: Maximum number of steps before truncating.
            top_k: The number of document chunks to retrieve from RAGFlow.
        """
        self.max_iter = max_iter
        self.dataset_ids = dataset_ids
        self.top_k = top_k
        
        ragflow_api_key = os.getenv("RAGFLOW_API_KEY")
        if not ragflow_api_key:
            # This check is important for live runs, but will be mocked in unit tests
            pass
            
        self.ragflow_client = RAGFlow(api_key=ragflow_api_key, base_url=ragflow_base_url)
        self.state: State | None = None
        self.curr_iter: int = 0

    def reset(self, question: str) -> Tuple[State, Dict[str, Any]]:
        """Resets the environment to a new initial state."""
        self.curr_iter = 0
        self.state = State(question=question, history=History())
        return self.state, self._get_info()

    def step(self, action: Action) -> Tuple[State, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action: The action to perform (either Search or Finish).

        Returns:
            A tuple of (next_state, terminated, truncated, info).
        """
        if self.state is None:
            raise RuntimeError("Environment must be reset before calling step.")

        self.curr_iter += 1
        truncated = self.curr_iter >= self.max_iter
        
        next_state, terminated = self._transition(self.state, action, truncated)
        
        self.state = next_state
        return self.state, terminated, truncated, self._get_info()

    def _transition(self, state: State, action: Action, truncated: bool) -> Tuple[State, bool]:
        """Calculates the next state based on the action."""
        history = state.history.copy()
        answer = None
        terminated = False

        # If the action has an answer, it's a Finish action.
        if action.answer:
            answer = action.answer
            terminated = True
        # If it has a query and we are not at the last step, it's a Search action.
        elif action.query and not truncated:
            documents = self._retrieve_documents(action.query)
            history.add_qd(query=action.query, documents=documents)

        next_state = State(question=state.question, history=history, answer=answer)
        return next_state, terminated

    def _retrieve_documents(self, query: str) -> List[str]:
        """Calls the RAGFlow API to retrieve document chunks."""
        try:
            # ragflow-sdk returns Chunk objects, we extract the content_str
            chunks = self.ragflow_client.retrieve(
                question=query,
                dataset_ids=self.dataset_ids,
                top_k=self.top_k,
            )
            return [chunk.content_str for chunk in chunks]
        except Exception as e:
            print(f"Error retrieving from RAGFlow: {e}")
            return []

    def _get_info(self) -> Dict[str, Any]:
        """Returns diagnostic information about the environment."""
        return {
            "curr_iter": self.curr_iter,
            "max_iter": self.max_iter,
            "dataset_ids": self.dataset_ids,
        }
```

-----

### \#\# 4. Implement Unit and Integration Tests

You will create two separate test files: one for fast, isolated unit tests (using mocks) and one for the live integration test.

#### A. Unit Test (with Mocks)

Create a new file at **`tests/test_retrieval_env.py`**. This test will verify the internal state logic without any network calls.

**File: `tests/test_retrieval_env.py`**

```python
import unittest
from unittest.mock import patch, MagicMock

from rag_gym_impl.upstream import import_symbols
Action = import_symbols("rag_gym.envs.action", "Action")[0]
from rag_gym_impl.retrieval_env import RetrievalEnv

# We patch the RAGFlow class within the module where it's imported and used.
@patch('rag_gym_impl.retrieval_env.RAGFlow')
class TestRetrievalEnv(unittest.TestCase):

    def test_step_with_search_action(self, MockRAGFlow: MagicMock):
        """Verify that a search action calls the mocked client and updates state."""
        # 1. Arrange
        # The mock instance is created when RetrievalEnv is initialized
        mock_ragflow_instance = MockRAGFlow.return_value
        mock_ragflow_instance.retrieve.return_value = [MagicMock(content_str="mocked doc 1")]

        env = RetrievalEnv(dataset_ids=["test_id"])
        state, _ = env.reset(question="test question")
        search_action = Action(query="test query")
        
        # 2. Act
        next_state, terminated, truncated, _ = env.step(search_action)

        # 3. Assert
        # Check that the mock retrieve method was called correctly
        mock_ragflow_instance.retrieve.assert_called_once_with(
            question="test query",
            dataset_ids=["test_id"],
            top_k=5
        )
        # Check that the state history was updated
        self.assertEqual(len(next_state.history), 1)
        self.assertEqual(next_state.history[0]["documents"], ["mocked doc 1"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_step_with_finish_action(self, MockRAGFlow: MagicMock):
        """Verify that a finish action terminates the episode."""
        env = RetrievalEnv(dataset_ids=["test_id"])
        state, _ = env.reset(question="test question")
        finish_action = Action(answer="final answer")
        
        next_state, terminated, truncated, _ = env.step(finish_action)
        
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(next_state.answer, "final answer")

    def test_truncation_after_max_iter(self, MockRAGFlow: MagicMock):
        """Verify the environment truncates after max_iter steps."""
        env = RetrievalEnv(dataset_ids=["test_id"], max_iter=1)
        state, _ = env.reset(question="test question")
        action = Action(query="test query")
        
        next_state, terminated, truncated, _ = env.step(action)

        self.assertFalse(terminated)
        self.assertTrue(truncated)

```

#### B. Integration Test (Live Connection)

Create a new file at **`tests/test_retrieval_env_integration.py`**. This confirms that the connection and data parsing from the real RAGFlow service works.

**File: `tests/test_retrieval_env_integration.py`**

```python
import pytest
import os
import time
from dotenv import load_dotenv

from ragflow_sdk import RAGFlow
from rag_gym_impl.upstream import import_symbols
Action = import_symbols("rag_gym.envs.action", "Action")[0]
from rag_gym_impl.retrieval_env import RetrievalEnv

@pytest.fixture(scope="module")
def ragflow_setup():
    """
    Sets up a test dataset in RAGFlow, yields the client and dataset ID,
    and cleans up by deleting the dataset afterward.
    """
    load_dotenv()
    api_key = os.getenv("RAGFLOW_API_KEY")
    base_url = "http://ragflow-server:9380"
    
    if not api_key:
        pytest.skip("RAGFLOW_API_KEY not found, skipping integration test.")
    
    rag_client = RAGFlow(api_key=api_key, base_url=base_url)
    
    dataset_name = f"test_dataset_{int(time.time())}"
    print(f"\nCreating test dataset: {dataset_name}")
    dataset = rag_client.create_dataset(name=dataset_name)
    
    try:
        doc_content = "The first man on the moon was Neil Armstrong."
        documents_to_upload = [{"display_name": "test_doc.txt", "blob": doc_content.encode('utf-8')}]
        dataset.upload_documents(documents_to_upload)
        time.sleep(5)
        yield rag_client, dataset.id
    finally:
        print(f"\nDeleting test dataset: {dataset_name}")
        rag_client.delete_dataset(dataset_id=dataset.id)

@pytest.mark.integration
class TestRetrievalEnvIntegration:
    def test_step_with_search_action(self, ragflow_setup):
        """
        Tests the step function with a real RAGFlow backend.
        """
        rag_client, dataset_id = ragflow_setup
        env = RetrievalEnv(dataset_ids=[dataset_id])
        state, _ = env.reset(question="Who was the first man on the moon?")
        search_action = Action(query="first person on the moon")

        next_state, terminated, truncated, info = env.step(search_action)

        assert not terminated
        retrieved_docs = next_state.history[0].get("documents", [])
        print(f"Retrieved documents: {retrieved_docs}")
        assert len(retrieved_docs) > 0
        assert "Neil Armstrong" in retrieved_docs[0]
```

-----

### \#\# 5. Run, Validate, and Document

Follow the established Docker workflow to test your implementation.

1.  **Enter the Docker container:**

    ```bash
    ./docker-exec.sh
    ```

2.  **Inside the container, run the tests.**

    ```bash
    cd /app/rag-gym-impl
    poetry install --no-root --no-interaction

    # Run only the fast unit tests
    PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -m "not integration" -sv

    # Run only the slow integration test
    PYTHONPATH=/app/rag-gym-impl/src poetry run pytest -m "integration" -sv
    ```

3.  **Iterative Debugging:** Run the tests. If any fail, analyze the output, fix the code, and rerun the tests. Repeat this cycle until all tests pass.

4.  **Update Documentation:** Once all tests are passing, update the **`README.md`** file. Modify the "How this differs..." and "Current status" sections to reflect that the custom `RetrievalEnv` is now implemented and connected to RAGFlow, completing the core agent architecture.
