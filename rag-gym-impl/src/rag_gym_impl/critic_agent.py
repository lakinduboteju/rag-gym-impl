import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from .upstream import import_symbols

State = import_symbols("rag_gym.envs.state", "State")[0]
Action = import_symbols("rag_gym.envs.action", "Action")[0]


class CriticAgent:
    """
    An agent that uses an LLM to select the single best action from a list of
    candidates, given the current problem-solving state.
    """

    def __init__(self, model_name: str = "gpt-5-nano"):
        """Initializes the Critic agent and its LangChain chain."""
        load_dotenv()

        system_prompt = (
            "You are a meticulous and strategic research analyst. Your task is to evaluate a list of proposed next actions and select the single most effective one to advance the research goal.\n\n"
            "Analyze the original question, the history of actions taken, and the retrieved information. Then, review the list of candidate actions. Choose the one action that will most directly and efficiently lead to the final answer.\n\n"
            "You MUST format your response as a single JSON object representing your chosen action. Do not include any other text, reasoning, or explanations. Just the single JSON object.\n\n"
            "Example Input:\n"
            "{{\n    \\\"question\\\": \\\"...\\\",\n    \\\"history\\\": [...],\n    \\\"candidate_actions\\\": [\n        {{\\\"action\\\": \\\"Search\\\", \\\"query\\\": \\\"query A\\\"}},\n        {{\\\"action\\\": \\\"Search\\\", \\\"query\\\": \\\"query B\\\"}},\n        {{\\\"action\\\": \\\"Finish\\\", \\\"answer\\\": \\\"...\\\"}}\n    ]\n}}\n\n"
            "Example Output (if query A is best):\n"
            "{{\\\"action\\\": \\\"Search\\\", \\\"query\\\": \\\"query A\\\"}}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the current state and the candidate actions:\n\n```json\n{input_json}\n```"),
            ]
        )
        model = ChatOpenAI(model=model_name, temperature=0.0)
        output_parser = JsonOutputParser()
        self.chain = prompt | model | output_parser

    def _format_input_for_prompt(self, state: State, actions: List[Action]) -> Dict[str, Any]:
        """Formats the state and candidate actions into a single dictionary for the LLM."""
        history_list: List[Dict[str, Any]] = []
        for item in state.history:
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

        candidate_actions_list: List[Dict[str, str]] = []
        for act in actions:
            if getattr(act, "query", None):
                candidate_actions_list.append({"action": "Search", "query": act.query})
            elif getattr(act, "answer", None):
                candidate_actions_list.append({"action": "Finish", "answer": act.answer})

        return {
            "question": state.question,
            "history": history_list,
            "candidate_actions": candidate_actions_list,
        }

    def _parse_llm_output_to_action(self, llm_output: Dict[str, str]) -> Action:
        """Converts the LLM's dictionary output into a single Action object."""
        action_type = llm_output.get("action")
        if action_type == "Search":
            return Action(query=llm_output.get("query", ""))
        if action_type == "Finish":
            return Action(answer=llm_output.get("answer", ""))
        return Action()

    def select_action(self, state: State, actions: List[Action]) -> Action:
        """
        Selects the best action from a list of candidates.

        Args:
            state: The current State object.
            actions: A list of candidate Action objects from the Actor.
        Returns:
            The single best Action object as chosen by the LLM.
        """
        formatted_input = self._format_input_for_prompt(state, actions)
        input_json_string = json.dumps(formatted_input, indent=2)

        llm_output = self.chain.invoke({"input_json": input_json_string})

        return self._parse_llm_output_to_action(llm_output)


