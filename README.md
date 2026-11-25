# Agentic Chatbot using LangGraph

A Streamlit chat app backed by a LangGraph agent running on Google Gemini. The agent can search the web, check the weather, look up stock prices, do arithmetic, and answer questions about a PDF you upload. Tools that call external services pause and ask you to approve them before they run. Conversations are saved in SQLite, so you can return to them later.

## Features

- **Tool-calling agent**: a LangGraph `StateGraph` loops between the LLM and a `ToolNode` until the model has a final answer.
- **Human in the loop**: `web_search`, `get_weather` and `get_stock_price` pause the graph with `interrupt()` and show an Approve / Reject card. A rejected call returns a message telling the model to answer without that tool.
- **Chat with a PDF (RAG)**: attach a PDF in the chat input. It is split into chunks, embedded with `gemini-embedding-001`, and indexed in FAISS. The `rag_tool` returns the 4 most similar chunks with their page numbers.
- **Saved conversations**: a `SqliteSaver` checkpointer writes every thread to `chatbot.db`. The sidebar lists past chats, titled by their first question.
- **Streaming UI**: replies stream token by token, and a status box shows which tools are running.

## Tools

| Tool | What it does | Needs approval | API key |
| --- | --- | --- | --- |
| `web_search` | Tavily web search (2 results, advanced depth) | Yes | `TAVILY_API_KEY` |
| `get_weather` | Current weather for a city (Weatherstack) | Yes | `WEATHERSTACK_API_KEY` |
| `get_stock_price` | Latest quote for a ticker (Alpha Vantage `GLOBAL_QUOTE`) | Yes | `ALPHAVANTAGE_API_KEY` |
| `calculator` | `add`, `sub`, `mul` or `div` on two numbers | No | none |
| `rag_tool` | Searches the uploaded PDF | No | `GOOGLE_API_KEY` (embeddings) |

## How it works

```
START ──► chat_node ──(tool call?)──► tools ──► chat_node ──► … ──► END
                         │
                         └──(no tool call)──► END
```

- `chat_node` adds a system prompt and calls Gemini (`gemini-3.5-flash-lite`) with the tools bound.
- `tools_condition` sends the turn to `tools` if the model asked for one, otherwise ends it.
- When an approval-gated tool runs, it calls `interrupt()`. The graph stops, `app.py` shows the approval card, and the chat input is disabled until you decide. Your choice resumes the graph with `Command(resume=...)`.

## Project structure

```
├── app.py            # Streamlit UI: chat, sidebar threads, PDF upload, approval card
├── backend.py        # LLM, tools, RAG indexing, LangGraph graph, SQLite checkpointer
├── requirements.txt
└── LICENSE
```

## Setup

1. Clone the repo and create a virtual environment:

   ```bash
   git clone <repo-url>
   cd Agentic-Chatbot-using-LangGraph
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   WEATHERSTACK_API_KEY=your_weatherstack_api_key
   ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
   ```

   Get keys from [Google AI Studio](https://aistudio.google.com/), [Tavily](https://tavily.com/), [Weatherstack](https://weatherstack.com/) and [Alpha Vantage](https://www.alphavantage.co/). Only `GOOGLE_API_KEY` is needed to start the app; a tool whose key is missing will fail when it's called.

4. Run the app:

   ```bash
   streamlit run app.py
   ```

## Usage

- Type a question in the chat box. Examples:
  - "What's the weather in Hyderabad?" → approve the `get_weather` call
  - "What is AAPL trading at?" → approve the `get_stock_price` call
  - "What is 1234 * 5.6?" → runs `calculator` straight away
- To ask about a document, attach a PDF with the chat input's attach button, then ask about it. The answer cites page numbers.
- Click **➕ New Chat** to start a new thread, or pick an earlier conversation from **My Conversations** in the sidebar.

## Notes and limitations

- The PDF index is kept in memory and shared by every chat. It is lost when the app restarts, and uploading a new PDF replaces the previous one.
- Chat history is stored in `chatbot.db` in the project folder. Delete that file to clear all conversations.
- If an uploaded PDF shows as 0 bytes (for example, a cloud-only OneDrive file), download it locally and upload it again.

## License

Licensed under the [Apache License 2.0](LICENSE).
