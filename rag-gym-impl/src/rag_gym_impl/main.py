from rag_gym_impl.upstream import import_symbol

State = import_symbol("rag_gym.envs.state", "State")


def main():
    state = State(question="What is RAG-Gym?")
    print(state.return_as_json())


if __name__ == "__main__":
    main()
