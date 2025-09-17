import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]
Action = import_symbols("rag_gym.envs.action", "Action")[0]


class ActorAgent:
    """
    An agent that uses an LLM to generate a set of candidate actions
    based on the current problem-solving state.
    """

    def __init__(self, model_name: str = "gpt-5-nano"):
        """Initializes the Actor agent and its LangChain chain."""
        load_dotenv()

        system_prompt = (
            "You are an expert research assistant. Your goal is to help answer a complex question by proposing the next best actions.\n"
            "Based on the user's question and the history of previous actions and retrieved documents, generate a list of 3 to 5 potential next actions.\n"
            "The valid actions are:\n"
            "1. `Search(query: str)`: To search for new information. The query should be specific and targeted to fill knowledge gaps.\n"
            "2. `Finish(answer: str)`: To conclude the research and provide a final answer. Only use this if you are confident you have enough information.\n\n"
            "You MUST format your response as a valid JSON list of objects. Do not include any other text, just the JSON.\n"
            "Example:\n"
            "[\n  {{\"action\": \"Search\", \"query\": \"specific details about topic X\"}},\n  {{\"action\": \"Finish\", \"answer\": \"The final answer is...\"}}\n]"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the current state:\n\n```json\n{state_json}\n```"),
            ]
        )
        model = ChatOpenAI(model=model_name, temperature=0.7)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_state_for_prompt(self, state: State) -> Dict[str, Any]:
        """Converts the State object into a JSON-serializable dictionary."""
        history_list: List[Dict[str, Any]] = []

        # Upstream History is iterable over dicts with keys query/documents
        for item in state.history:
            # Expecting dicts like {"query": str, "documents": list[...]} per vendored State
            query = item.get("query") if isinstance(item, dict) else None
            documents = item.get("documents") if isinstance(item, dict) else None
            if query is not None and documents is not None:
                history_list.append(
                    {
                        "action": "Search",
                        "query": query,
                        "document_chunks": documents,
                    }
                )

        return {
            "question": state.question,
            "history": history_list,
        }

    def _parse_llm_output_to_actions(self, llm_output: List[Dict[str, str]]) -> List[Action]:
        """Converts the LLM's dictionary output into a list of Action objects."""
        actions: List[Action] = []
        for item in llm_output:
            action_type = item.get("action")
            if action_type == "Search":
                actions.append(Action(query=item.get("query", "")))
            elif action_type == "Finish":
                actions.append(Action(answer=item.get("answer", "")))
        return actions

    def generate_actions(self, state: State) -> List[Action]:
        """
        Generates candidate actions for a given state.

        Args:
            state: The current State object from the RAG-Gym environment.
        Returns:
            A list of candidate Action objects.
        """
        formatted_state = self._format_state_for_prompt(state)
        state_json_string = json.dumps(formatted_state, indent=2)

        llm_output = self.chain.invoke({"state_json": state_json_string})

        return self._parse_llm_output_to_actions(llm_output)


