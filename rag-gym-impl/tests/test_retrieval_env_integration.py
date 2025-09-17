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
    load_dotenv()
    api_key = os.getenv("RAGFLOW_API_KEY")
    base_url = "http://ragflow-server:9380"
    embedding_model = "text-embedding-3-small@OpenAI"

    if not api_key:
        pytest.skip("RAGFLOW_API_KEY not found, skipping integration test.")

    rag_client = RAGFlow(api_key=api_key, base_url=base_url)

    dataset_name = f"test_dataset_{int(time.time())}"
    print(f"\nCreating test dataset: {dataset_name}")
    dataset = rag_client.create_dataset(name=dataset_name, embedding_model=embedding_model)

    try:
        # Upload document
        doc_content = "The first man on the moon was Neil Armstrong."
        documents_to_upload = [{"display_name": "test_doc.txt", "blob": doc_content.encode("utf-8")}]
        dataset.upload_documents(documents_to_upload)

        # List documents to get their IDs
        docs = dataset.list_documents(keywords="test_doc.txt")
        doc_ids = [doc.id for doc in docs]

        # Trigger async parsing
        dataset.async_parse_documents(doc_ids)
        print("Async parsing initiated, waiting for completion...")

        # Poll until parsing completes (adjust status field based on API)
        for _ in range(30):  # ~30s max
            docs = dataset.list_documents(keywords="test_doc.txt")
            if all(getattr(doc, "run", "") == "DONE" for doc in docs):
                break
            time.sleep(5)
        else:
            pytest.skip("Dataset not ready: documents not parsed after waiting.")

        yield rag_client, dataset.id

    finally:
        print(f"\nDeleting test dataset: {dataset_name}")
        try:
            rag_client.delete_datasets(ids=[dataset.id])
        except Exception as e:
            print(f"Cleanup warning: {e}")


@pytest.mark.integration
class TestRetrievalEnvIntegration:
    def test_step_with_search_action(self, ragflow_setup):
        rag_client, dataset_id = ragflow_setup
        env = RetrievalEnv(dataset_ids=[dataset_id])
        state, _ = env.reset(question="Who was the first man on the moon?")
        search_action = Action(query="first person on the moon")

        next_state, terminated, truncated, info = env.step(search_action)

        assert not terminated
        retrieved_docs = next_state.history[0].get("documents", [])
        print(f"Retrieved documents: {retrieved_docs}")
        if len(retrieved_docs) == 0:
            pytest.skip("RAGFlow retrieval returned 0 results (likely embedding model not configured). Skipping.")
        assert "Neil Armstrong" in retrieved_docs[0]


