from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.types import interrupt
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import requests
import os
import sqlite3

load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=GOOGLE_API_KEY
)

# ---------------- RAG: PDF -> chunks -> embeddings -> FAISS ----------------
embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', api_key=GOOGLE_API_KEY)

retriever = None   # set when a PDF is ingested


def ingest_rag_document(file_path: str):
    """Load a PDF, split it, embed it, and keep a retriever for searching it."""
    global retriever
    loader = PyPDFLoader(file_path)
    docs = loader.load()                                   # one Document per page

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)                # smaller overlapping pieces

    vector_store = FAISS.from_documents(chunks, embeddings)  # embed + index
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    return len(chunks)


def get_retrieved(query: str):
    """Return the chunks most similar to the query."""
    if retriever is None:
        return []
    return retriever.invoke(query)

# ---------------- human in the loop (called inside a tool) ----------------
REJECTED = {'status': 'rejected', 'message': 'The user rejected this tool call. Do not retry it; answer without it.'}


def ask_approval(tool_name: str, args: dict) -> bool:
    """Pause the graph inside the tool and wait for the user's decision."""
    decision = interrupt({'tool_calls': [{'name': tool_name, 'args': args}]})   # UI resumes with 'approve' / 'reject'
    return decision == 'approve'


# ---------------- tools ----------------
tavily = TavilySearch(max_results=2, topic='general', search_depth='advanced', tavily_api_key=TAVILY_API_KEY)


@tool
def web_search(query: str) -> dict:
    """Search the web for current information, news and facts."""
    if not ask_approval('web_search', {'query': query}):
        return REJECTED
    return tavily.invoke({'query': query})


@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Do basic math on two numbers. operation is one of: add, sub, mul, div."""
    if operation == 'add':
        return a + b
    if operation == 'sub':
        return a - b
    if operation == 'mul':
        return a * b
    if operation == 'div':
        return a / b
    return 'unknown operation'


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    if not ask_approval('get_weather', {'city': city}):
        return REJECTED
    response = requests.get(
        'http://api.weatherstack.com/current',
        params={'access_key': WEATHERSTACK_API_KEY, 'query': city}
    )
    return response.json()


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock price for a ticker symbol, e.g. 'AAPL', 'TSLA', 'MSFT'."""
    if not ask_approval('get_stock_price', {'symbol': symbol}):
        return REJECTED
    response = requests.get(
        'https://www.alphavantage.co/query',
        params={'function': 'GLOBAL_QUOTE', 'symbol': symbol, 'apikey': ALPHAVANTAGE_API_KEY}
    )
    return response.json()


@tool
def rag_tool(query: str) -> dict:
    """Search the uploaded PDF. Use it for any question about the uploaded document."""
    results = get_retrieved(query)
    if not results:
        return {'error': 'No document uploaded yet. Ask the user to upload a PDF first.'}
    return {
        'query': query,
        'context': [doc.page_content for doc in results],
        'pages': [doc.metadata.get('page') for doc in results],
    }


tools = [web_search, calculator, get_weather, get_stock_price, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# ---------------- graph ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant. "
    "For questions about the user's uploaded PDF, call rag_tool and answer only from its context, mentioning page numbers. "
    "Use the other tools for web search, math, weather and stock prices."
))


def chatnode(state: ChatState):
    messages = [SYSTEM_PROMPT] + state["messages"]   # system prompt is not saved to history
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chatnode)
graph.add_node('tools', ToolNode(tools))

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)   # tool call? -> tools, else -> END
graph.add_edge('tools', 'chat_node')                        # tool result back to the LLM

workflow = graph.compile(checkpointer=checkpoint)


def get_pending_approval(config):
    """Return the first paused tool request for this thread (with its interrupt id), or None."""
    state = workflow.get_state(config)
    if state.interrupts:
        first = state.interrupts[0]
        return {'id': first.id, **first.value}   # {'id': ..., 'tool_calls': [...]}
    return None


def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)
