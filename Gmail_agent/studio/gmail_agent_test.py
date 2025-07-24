
import os
from configuration import AgentConfigurable
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.errors import NodeInterrupt
from dotenv import load_dotenv
from langchain_arcade import ToolManager

# ---- LOAD ENVIRONMENT VARIABLES ---- #
load_dotenv()
arcade_api_key = os.getenv("ARCADE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not arcade_api_key:
    raise ValueError("ARCADE_API_KEY is not set")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")

# ---- INIT ARCADE TOOL MANAGER AND GMAIL TOOLS ---- #
manager = ToolManager(api_key=arcade_api_key)
gmail_tools = manager.init_tools(toolkits=["Gmail"])


class AgentState(MessagesState):
    auth_url: str | None = None

###### ---- NODES ---- ######

def check_auth(state: AgentState, config: dict):
    """Checks if the user is authenticated."""
    user_id = config["configurable"].get("user_id")
    tool_name = state["messages"][-1].tool_calls[0]["name"]
    auth_response = manager.authorize(tool_name, user_id)
    print(auth_response)
    if auth_response.status != "completed":
        return {"auth_url": auth_response.url}
    else:
        return {"auth_url": None}

def authorize(state: MessagesState, config: dict):
    """Authorizes the user."""
    user_id = config["configurable"].get("user_id")
    tool_name = state["messages"][-1].tool_calls[0]["name"]
    auth_response = manager.authorize(tool_name, user_id)
    if auth_response.status != "completed":
        auth_message = (
            f"Please authorize the application in your browser:\n\n {state.get('auth_url')}"
        )
        raise NodeInterrupt(auth_message + str(state.get("auth_url")))

# Should Continue #
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "check_auth"
    return END

# ---- TOOLS LIST ---- #
tools = gmail_tools

# ---- LLM WITH TOOLS ---- #
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# ---- SYSTEM MESSAGE ---- #
sys_msg = SystemMessage(content="You are a helpful assistant that can use tools to help users with tasks.")

# ---- ASSISTANT NODE ---- #
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# ---- BUILD GRAPH ---- #
builder = StateGraph(AgentState, AgentConfigurable)
builder.add_node("agent", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("check_auth", check_auth)
builder.add_node("authorize", authorize)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    ["check_auth", END]
)

builder.add_edge("check_auth", "authorize")
builder.add_edge("authorize", "tools")
builder.add_edge("tools", "agent")

graph = builder.compile()

# ---- MAIN LOOP (for local testing) ---- #
if __name__ == "__main__":
    print("Jasper Gmail Agent: Ready to help you with your emails!")
    state = {"messages": []}
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"no", "bye", "exit", "quit", "", "thanks"}:
            print("Jasper: Goodbye!")
            break
        state["messages"].append({"content": user_input, "role": "user"})
        state = graph.invoke(state)
        assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
        if assistant_msgs:
            print(f"Jasper: {assistant_msgs[-1]['content']}")

