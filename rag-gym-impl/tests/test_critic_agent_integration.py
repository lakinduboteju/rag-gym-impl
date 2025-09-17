import pytest
import os
from dotenv import load_dotenv

from rag_gym_impl.upstream import import_symbols

State, History = import_symbols("rag_gym.envs.state", "State", "History")
Action = import_symbols("rag_gym.envs.action", "Action")[0]

from rag_gym_impl.critic_agent import CriticAgent


@pytest.mark.integration
class TestCriticAgentIntegration:

    def test_select_action_for_multihop_question(self):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not found, skipping integration test.")

        agent = CriticAgent()

        state = State(
            question="What is the date of death of the director of the film Holocaust 2000?",
            history=History(),
        )
        state.history.add_qd(
            query="Who directed the film Holocaust 2000?",
            documents=["The film was directed by Alberto De Martino."],
        )

        candidate_actions = [
            Action(query="Alberto De Martino filmography"),
            Action(query="Alberto De Martino date of death"),
            Action(answer="The director of Holocaust 2000 is Alberto De Martino."),
        ]

        best_action = agent.select_action(state, candidate_actions)

        print(f"\nChosen action: Query='{best_action.query}', Answer='{best_action.answer}'")
        assert isinstance(best_action, Action)
        assert getattr(best_action, "query", None)
        assert "death" in best_action.query.lower()
        assert "alberto de martino" in best_action.query.lower()


