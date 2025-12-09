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
