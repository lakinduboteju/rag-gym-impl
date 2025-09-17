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


