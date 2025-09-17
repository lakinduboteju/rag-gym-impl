import os
from typing import Tuple, Dict, Any, List

from ragflow_sdk import RAGFlow

from .upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]
Action = import_symbols("rag_gym.envs.action", "Action")[0]
History = import_symbols("rag_gym.envs.state", "History")[0]


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
        # API key may be missing in tests; the client will fail only on call
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

        if getattr(action, "answer", None):
            answer = action.answer
            terminated = True
        elif getattr(action, "query", None) and not truncated:
            documents = self._retrieve_documents(action.query)
            history.add_qd(query=action.query, documents=documents)

        next_state = State(question=state.question, history=history, answer=answer)
        return next_state, terminated

    def _retrieve_documents(self, query: str) -> List[str]:
        """Calls the RAGFlow API to retrieve document chunks."""
        try:
            chunks = self.ragflow_client.retrieve(
                question=query,
                dataset_ids=self.dataset_ids,
                top_k=self.top_k,
            )
            return [chunk.content for chunk in chunks]
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


