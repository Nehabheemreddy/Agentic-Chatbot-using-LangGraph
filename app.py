from backend import workflow, get_all_threads, ingest_rag_document, get_pending_approval
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
import streamlit as st
import tempfile
import uuid
import os


# ---------------- helpers ----------------
def generate_thread_id():
    return str(uuid.uuid4())


def get_config(thread_id):
    return {
        'configurable': {'thread_id': thread_id},
        'metadata': {'thread_id': thread_id},
        'run_name': 'langgraph-streamlit'}


def start_new_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    st.session_state['pending_approval'] = None
    st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    # read saved messages for this thread from the LangGraph checkpointer
    state = workflow.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    history = []
    for msg in messages:
        if isinstance(msg, ToolMessage) or not msg.text:   # skip raw tool output and empty tool-call messages
            continue
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        history.append({'role': role, 'content': msg.text})
    return history


def thread_title(thread_id):
    # use the first user question as the sidebar label
    history = load_conversation(thread_id)
    for msg in history:
        if msg['role'] == 'user':
            return msg['content'][:30]
    return 'New chat'


def run_graph(graph_input):
    """Run the graph (new message or approval decision), stream the reply, and check for a new approval request."""
    config = get_config(st.session_state['thread_id'])

    with st.chat_message('assistant'):
        status_holder = {'box': None, 'used': []}   # tool progress box + names of tools used

        def ai_only_stream():
            for message_chunk, metadata in workflow.stream(graph_input, config=config, stream_mode='messages'):
                # LLM asked for a tool -> show it in the progress box
                tool_calls = getattr(message_chunk, 'tool_call_chunks', None) or []
                for call in tool_calls:
                    if call.get('name'):
                        status_holder['used'].append(call['name'])
                        label = f"🔧 Using `{call['name']}` …"
                        if status_holder['box'] is None:
                            status_holder['box'] = st.status(label, expanded=True)
                        else:
                            status_holder['box'].update(label=label, state='running', expanded=True)

                # tool result arrived -> don't print raw output
                if isinstance(message_chunk, ToolMessage):
                    continue

                # only stream the LLM's text
                if isinstance(message_chunk, AIMessage) and message_chunk.text:
                    yield message_chunk.text

        ai_message = st.write_stream(ai_only_stream())

        # graph paused for approval?
        pending = get_pending_approval(config)
        st.session_state['pending_approval'] = pending

        if status_holder['box'] is not None:
            used = ', '.join(f"`{name}`" for name in status_holder['used'])
            if pending:
                status_holder['box'].update(label=f"⏸️ Waiting for approval: {used}", state='running', expanded=False)
            else:
                status_holder['box'].update(label=f"✅ Used {used}", state='complete', expanded=False)

    if isinstance(ai_message, str) and ai_message.strip():
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})


# ---------------- session setup ----------------
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'thread_id' not in st.session_state:
    start_new_chat()


# ---------------- sidebar ----------------
st.sidebar.title('Agentic Chatbot')

if st.sidebar.button('➕ New Chat'):
    start_new_chat()

if st.session_state.get('ingested_file'):
    st.sidebar.caption(f"📄 Using document: **{st.session_state['ingested_file']}**")

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:   # newest first
    if st.sidebar.button(thread_title(thread_id), key=thread_id):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_conversation(thread_id)
        st.session_state['pending_approval'] = get_pending_approval(get_config(thread_id))   # chat may be paused


# ---------------- main chat ----------------
st.title("Agentic Chatbot with Langgraph")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


# ---------------- human in the loop: approval card ----------------
pending = st.session_state.get('pending_approval')
if pending:
    with st.chat_message('assistant'):
        st.warning('🙋 The assistant wants to run these tools. Allow it?')
        for call in pending['tool_calls']:
            args = ', '.join(f'{k}={v!r}' for k, v in call['args'].items())
            st.code(f"{call['name']}({args})", language='python')
        col_approve, col_reject = st.columns(2)
        approve = col_approve.button('✅ Approve', use_container_width=True)
        reject = col_reject.button('❌ Reject', use_container_width=True)

    if approve or reject:
        st.session_state['pending_approval'] = None
        decision = 'approve' if approve else 'reject'
        run_graph(Command(resume={pending['id']: decision}))   # continue the paused tool
        st.rerun()


# ---------------- chat input ----------------
chat_value = st.chat_input(
    'Approve or reject the tool request first…' if pending else 'type here..',
    accept_file=True, file_type=['pdf'],
    disabled=bool(pending))   # no new messages while waiting for approval

user_input = None
if chat_value:
    user_input = chat_value.text

    # ---- PDF attached -> index it for RAG ----
    for uploaded_pdf in chat_value.files:
        if not uploaded_pdf.getvalue():   # 0-byte file (e.g. cloud-only OneDrive file)
            st.error(f'"{uploaded_pdf.name}" is empty (0 bytes). Open/download it locally, then upload again.')
            continue
        with st.status(f'Indexing {uploaded_pdf.name}…', expanded=True) as status_box:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_pdf.getvalue())     # PyPDFLoader needs a file path
                temp_path = temp_file.name
            try:
                num_chunks = ingest_rag_document(temp_path)
            finally:
                os.remove(temp_path)
            st.session_state['ingested_file'] = uploaded_pdf.name
            status_box.update(label=f'✅ Indexed {uploaded_pdf.name} ({num_chunks} chunks)', state='complete', expanded=False)

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    run_graph({'messages': [HumanMessage(content=user_input)]})
    if st.session_state.get('pending_approval'):
        st.rerun()   # show the approval card
