from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Create tools - Arxiv
api_wrapper_arxiv =  ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arXiv papers")
print(arxiv.name)

print("ArXiv Results ::: ",arxiv.invoke("Quantum Architecture Search")) # this is one of the research paper

# Create tools - Wikipedia
api_wrapper_wikipedia =  WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)
print(wiki.name)

print("WikiPedia Result::: ",wiki.invoke("About India")) # this is one of the research paper

#  Now my above two tools been tested and its ready for integration

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

### Tavily Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults

tavily = TavilySearchResults()

print("Tavily Results:::",tavily.invoke("latest tamilnadu news"))

### Next Step - We will be combiing all these tools - ArXiv, Wikipedia and Tavily

tools = [arxiv, wiki, tavily]

## Initialize the LLM models
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

print("LLM Results:::",llm.invoke("Who is Modi"))

### Our aim is combined all the tool with LLM
#https://www.youtube.com/watch?v=HCSPIH3I-vc&list=PLZoTAELRMXVPFd7JdvB-rnTb_5V26NYNO&index=3


#  How do we combine all the tools
llm_with_tools = llm.bind_tools(tools=tools)

# Execue the LLM call
print("LLM calls to specific tool:::  ",llm_with_tools.invoke("latest rearch in quantum"))

# ReAct(Reasoning Act)Architecture

# Create a workflow using langgraph
from typing_extensions import TypedDict # maintain entire state
from langchain_core.messages import AnyMessage # AI Message or human message
from typing import Annotated # labelling
from langgraph.graph.message import add_messages # Reducers in langgraph

# Step1: Create a schema
from typing import TypedDict
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]# add messgae is reducer, it will NOT override the message - It will append the message



# Entire chatbot with langgraph
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node definition
def tool_calling_llm (state: State):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools=tools))


# Edges
builder.add_edge(START, "tool_calling_llm")
# ----- MORE IMPORTANT ------->
builder.add_conditional_edges("tool_calling_llm"
                              
    # if the latest meesage(result) from assistant is a tool call -> tools_condition routes to tool
    # if the latest meesage(result) from assistant is NOT a tool call -> tools_condition routes to end
                              ,tools_condition)

builder.add_edge("tools", END)


# Display Graph
import os

## Run the grpah, we need to compile the graph
graph_builder = builder.compile()

# 1. Get the binary PNG data
png_data = graph_builder.get_graph().draw_mermaid_png()

# 2. Define the output file path
output_filename = "langgraph_visualization.png"

# 3. Write the binary data to the file
with open(output_filename, "wb") as f:
    f.write(png_data)

print(f"Graph saved successfully to: {os.path.abspath(output_filename)}")



from langchain_core.messages import HumanMessage

# 1. Format the input string into a HumanMessage object, wrapped in a list
user_input_message_list = [HumanMessage(content="1706.03762")]

# 2. Invoke the graph with a DICTIONARY that uses the state key "messages"
messages = graph_builder.invoke({"messages": user_input_message_list})
print(messages)

# Pretty print is organized and readable
# import pprint
# print("\n--- pprint.pprint() ---")
# pprint.pprint(messages)

