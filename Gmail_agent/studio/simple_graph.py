from langgraph.graph import StateGraph, END, START
from typing import TypedDict

# ✅ 1. Define schema properly as a TypedDict
class State(TypedDict):
    input: str
    output: str

# ✅ 2. Define a simple node
def echo_node(state: State) -> State:
    user_text = state["input"]
    return {"output": f"You said: {user_text}"}

# ✅ 3. Build the graph
graph = StateGraph(State)  # <-- Pass the TypedDict, NOT a dict
graph.add_node("echo", echo_node)
graph.add_edge(START, "echo")
graph.add_edge("echo", END)

# ✅ 4. Compile
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"input": "Hello LangGraph!"})
    print(result)