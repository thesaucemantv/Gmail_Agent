from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

# ---- TOOL FUNCTIONS ---- #
def wiki_search(query: str) -> str:
    """Search Wikipedia for factual information."""
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki.run(query)

def chit_chat(query: str) -> str:
    """Casual conversation."""
    llm = ChatOpenAI(model="gpt-4o")
    reply = llm.invoke(f"You are a friendly AI. Respond casually to: {query}")
    return reply.content

def summarize(context: str) -> str:
    """Summarize a given context for the user."""
    llm = ChatOpenAI(model="gpt-4o")
    summary = llm.invoke(f"Summarize this for a user: {context}")
    return summary.content

# ---- TOOLS LIST ---- #
tools = [wiki_search, chit_chat, summarize]

# ---- LLM WITH TOOLS ---- #
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# ---- SYSTEM MESSAGE ---- #
sys_msg = SystemMessage(content="You are Aura, an autonomous universal reasoning agent. You are helpful, friendly, and can use tools to answer questions or chat.")

# ---- ASSISTANT NODE ---- #
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# ---- BUILD GRAPH ---- #
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

graph = builder.compile()

# ---- MAIN LOOP ---- #
if __name__ == "__main__":
    print("Aura: Hello! How can I help you today?")
    state = {"messages": []}
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"no", "bye", "exit", "quit", "", "thanks"}:
            print("Aura: Goodbye!")
            break
        state["messages"].append({"content": user_input, "role": "user"})
        state = graph.invoke(state)
        # Find the latest assistant message
        assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
        if assistant_msgs:
            print(f"Aura: {assistant_msgs[-1]['content']}")