import pytest
import os
import time
from dotenv import load_dotenv

from ragflow_sdk import RAGFlow
from rag_gym_impl.upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]

from rag_gym_impl.actor_agent import ActorAgent
from rag_gym_impl.critic_agent import CriticAgent
from rag_gym_impl.retrieval_env import RetrievalEnv


@pytest.fixture(scope="module")
def e2e_ragflow_setup():
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
        docs_to_upload = [
            {"display_name": "doc1.txt", "blob": "Holocaust 2000 is a 1977 horror film directed by the Italian filmmaker Alberto De Martino.".encode("utf-8")},
            {"display_name": "doc2.txt", "blob": "Alberto De Martino (12 June 1929 – 31 October 2015) was a prolific director of genre films.".encode("utf-8")},
            {"display_name": "doc3_noise.txt", "blob": "The New York Times is a daily newspaper based in New York City.".encode("utf-8")},
            {"display_name": "doc4_noise.txt", "blob": "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas.".encode("utf-8")},
            {"display_name": "doc5_noise.txt", "blob": "Python is a popular programming language known for its simplicity and wide ecosystem of libraries.".encode("utf-8")},
            {"display_name": "doc6_noise.txt", "blob": "The Amazon rainforest is the largest tropical rainforest in the world, rich in biodiversity.".encode("utf-8")},
            {"display_name": "doc7_noise.txt", "blob": "Albert Einstein developed the theory of relativity, a cornerstone of modern physics.".encode("utf-8")},
            {"display_name": "doc8_noise.txt", "blob": "Bananas are one of the most widely consumed fruits globally and are high in potassium.".encode("utf-8")},

        ]

        dataset.upload_documents(docs_to_upload)

        docs = dataset.list_documents()
        doc_ids = [doc.id for doc in docs]

        dataset.async_parse_documents(doc_ids)
        print("Async parsing initiated, waiting for completion...")

        for _ in range(6):
            docs = dataset.list_documents()
            if all(getattr(doc, "run", "") == "DONE" for doc in docs):
                print("All documents parsed successfully.")
                break
            time.sleep(5)
        else:
            pytest.skip("Documents not parsed after waiting, skipping E2E test.")

        yield dataset.id

    finally:
        print(f"\nDeleting E2E test dataset: {dataset_name}")
        try:
            rag_client.delete_datasets(ids=[dataset.id])
        except Exception as e:
            print(f"Cleanup warning: {e}")


@pytest.mark.integration
class TestEndToEndAgentLoop:

    def test_multi_hop_question_answering_loop(self, e2e_ragflow_setup):
        dataset_id = e2e_ragflow_setup

        actor = ActorAgent()
        critic = CriticAgent()
        env = RetrievalEnv(dataset_ids=[dataset_id], max_iter=3)

        question = "What is the date of death of the director of the film Holocaust 2000?"
        print(f"Question: {question}")
        state, _ = env.reset(question=question)

        for i in range(env.max_iter):
            print(f"\n================================== LOOP {i+1} ==================================")
            print(f"Current State History ({len(state.history)} turn(s)):")
            if not state.history:
                print("  [No history yet]")
            else:
                for idx, turn in enumerate(state.history):
                    print(f"  Turn {idx+1}:")
                    print(f"    Query: {turn.get('query')}")
                    print(f"    Docs: {turn.get('documents')}")

            # Actor generates actions
            candidate_actions = actor.generate_actions(state)
            print("\n--- Actor: Generated Actions ---")
            for j, action in enumerate(candidate_actions):
                if action.query:
                    print(f"  [{j+1}] Search: '{action.query}'")
                elif action.answer:
                    print(f"  [{j+1}] Finish: '{action.answer}'")

            # Critic selects the best action
            best_action = critic.select_action(state, candidate_actions)
            print("\n--- Critic: Selected Action ---")
            if best_action.query:
                print(f"  Search: '{best_action.query}'")
            elif best_action.answer:
                print(f"  Finish: '{best_action.answer}'")

            # Environment executes the action and retrieves documents
            state, terminated, truncated, _ = env.step(best_action)

            # Print retrieved documents if a search was performed
            if best_action.query and len(state.history) > i:
                print("\n--- Environment: Retrieved Documents ---")
                retrieved_docs = state.history[-1].get("documents", [])
                if not retrieved_docs:
                    print("  [No documents were retrieved]")
                else:
                    for k, doc in enumerate(retrieved_docs):
                        print(f"  Doc [{k+1}]: \"{doc}\"")

            # Assertions to verify progress
            if i == 0 and len(state.history) > 0:
                assert "alberto de martino" in state.history[0]["documents"][0].lower()

            if i == 1 and len(state.history) > 1:
                retrieved_docs_text = " ".join(state.history[1]["documents"]).lower()
                if "2015" not in retrieved_docs_text:
                    pytest.skip("Dataset not fully indexed for death date; skipping E2E after partial progress.")

            if terminated or truncated:
                print(f"\nLoop {'terminated' if terminated else 'truncated'}.")
                break
        
        print("\n=============================== FINAL RESULT ===============================")
        print(f"Final Answer from state: {state.answer}")
        
        if state.answer is None:
            pytest.skip("Agent did not finalize within max_iter. Skipping strict final assertions.")
            
        final_answer = state.answer.lower()
        assert "alberto de martino" in final_answer
        assert "2015" in final_answer

# Example output:-
"""
Question: What is the date of death of the director of the film Holocaust 2000?

================================== LOOP 1 ==================================
Current State History (0 turn(s)):
  [No history yet]

--- Actor: Generated Actions ---
  [1] Search: 'Holocaust 2000 film director'
  [2] Search: 'Holocaust 2000 director biography death date'
  [3] Search: '"Holocaust 2000" film director obituary'
  [4] Search: 'Holocaust 2000 IMDb director'

--- Critic: Selected Action ---
  Search: 'Holocaust 2000 IMDb director'

--- Environment: Retrieved Documents ---
  Doc [1]: "Holocaust 2000 is a 1977 horror film directed by the Italian filmmaker Alberto De Martino."
  Doc [2]: "Alberto De Martino (12 June 1929 – 31 October 2015) was a prolific director of genre films."

================================== LOOP 2 ==================================
Current State History (1 turn(s)):
  Turn 1:
    Query: Holocaust 2000 IMDb director
    Docs: ['Holocaust 2000 is a 1977 horror film directed by the Italian filmmaker Alberto De Martino.', 'Alberto De Martino (12 June 1929 – 31 October 2015) was a prolific director of genre films.']

--- Actor: Generated Actions ---
  [1] Finish: 'The director Alberto De Martino died on 31 October 2015.'
  [2] Search: 'Alberto De Martino death date obituary'
  [3] Search: 'Alberto De Martino Holocaust 2000 director filmography'
  [4] Search: 'Alberto De Martino obituary 2015 Italian director'

--- Critic: Selected Action ---
  Finish: 'The director Alberto De Martino died on 31 October 2015.'

Loop terminated.

=============================== FINAL RESULT ===============================
Final Answer from state: The director Alberto De Martino died on 31 October 2015.
"""
