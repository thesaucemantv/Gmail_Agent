import os
from dotenv import load_dotenv
from langchain_arcade import ToolManager

load_dotenv()
manager = ToolManager(api_key=os.getenv("ARCADE_API_KEY"))
tools = manager.init_tools(toolkits=["Gmail"])

print(tools)