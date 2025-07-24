from langchain_arcade import ToolManager
from dotenv import load_dotenv
import os

load_dotenv()
arcade_api_key = os.getenv("ARCADE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

print(arcade_api_key)
print(openai_api_key)


if not arcade_api_key:
    raise ValueError("ARCADE_API_KEY is not set")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


manager = ToolManager(api_key=arcade_api_key)
 
# Fetch the "ScrapeUrl" tool from the "Firecrawl" toolkit
tools = manager.init_tools(tools=["Firecrawl.ScrapeUrl"])
print(manager.tools)
 
# Get all tools from the "Gmail" toolkit
tools = manager.init_tools(toolkits=["Gmail"])
print(manager.tools)