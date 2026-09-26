import os
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="NOVA AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing. Please check your .env file.")
    st.stop()

# =========================================================
# 3. CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 20%, rgba(124, 58, 237, 0.25), transparent 25%),
            radial-gradient(circle at 90% 10%, rgba(6, 182, 212, 0.20), transparent 25%),
            radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.15), transparent 25%),
            #070b1a;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(30, 27, 75, 0.98));
        border-right: 1px solid rgba(139, 92, 246, 0.25);
    }
    section[data-testid="stSidebar"] h1 { color: #ffffff; }

    .main-title {
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(90deg, #8b5cf6, #06b6d4, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #a5b4fc;
        font-size: 17px;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    .status-box {
        display: inline-flex;
        align-items: center;
        padding: 8px 15px;
        border-radius: 30px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #86efac;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .status-dot {
        width: 9px;
        height: 9px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 12px #22c55e;
    }

    .feature-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(6, 182, 212, 0.10));
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 18px;
        padding: 18px;
        min-height: 130px;
        transition: 0.3s;
    }
    .feature-card:hover {
        border-color: rgba(6, 182, 212, 0.7);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(6, 182, 212, 0.12);
    }
    .feature-icon { font-size: 30px; }
    .feature-title { font-size: 17px; font-weight: 700; margin-top: 8px; color: #ffffff; }
    .feature-text { font-size: 13px; color: #94a3b8; margin-top: 5px; }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.025);
        border-radius: 18px;
        padding: 12px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }

    [data-testid="stChatInput"] {
        border-radius: 18px;
        border: 1px solid rgba(139, 92, 246, 0.5);
        background: rgba(15, 23, 42, 0.9);
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.12);
    }
    [data-testid="stChatInput"] textarea { color: white !important; font-size: 16px; }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 30px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. LANGGRAPH LLM
# =========================================================
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)


# =========================================================
# 5. LANGGRAPH STATE
# =========================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================================================
# 6. CHAT NODE
# =========================================================
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# =========================================================
# 7. CREATE LANGGRAPH
# =========================================================
@st.cache_resource
def create_chatbot():
    memory = MemorySaver()
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=memory)


chatbot = create_chatbot()

# =========================================================
# 8. SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# 9. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;">
            <div style="font-size:55px; margin-bottom:5px;">✨</div>
            <h1 style="font-size:28px; margin-bottom:0px;">NOVA AI</h1>
            <p style="color:#94a3b8; font-size:13px;">Intelligent LangGraph Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🧠 AI Engine")
    st.info("Powered by LangGraph + Groq")

    st.markdown("### ⚡ Capabilities")
    st.markdown(
        """
        🧠 **AI Conversation**
        🔗 **LangGraph Workflow**
        💾 **Conversation Memory**
        ⚡ **Fast Groq Inference**
        💬 **Interactive Chat**
        """
    )

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        <div style="position:fixed; bottom:20px; color:#64748b; font-size:12px;">
        NOVA AI • 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 10. MAIN HEADER
# =========================================================
st.markdown(
    '<div class="status-box"><div class="status-dot"></div>AI Assistant Online</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="main-title">Meet NOVA AI ✨</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A smart conversational assistant powered by LangGraph and Groq.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# 11. FEATURE CARDS (shown only before the first message)
# =========================================================
if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)

    features = [
        ("🧠", "Smart AI", "Ask questions and get intelligent responses instantly."),
        ("🔗", "LangGraph", "Conversations are processed through a LangGraph workflow."),
        ("💾", "Memory", "Maintain context throughout your conversation."),
    ]

    for col, (icon, title, text) in zip((col1, col2, col3), features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# 12. CHAT HISTORY
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# 13. CHAT INPUT + PROCESSING
# =========================================================
user_input = st.chat_input("✨ Ask NOVA anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    config = {"configurable": {"thread_id": "streamlit-user"}}

    try:
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        ai_response = response["messages"][-1].content

        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

# =========================================================
# 14. FOOTER
# =========================================================
st.markdown(
    '<div class="footer">Built with ❤️ using LangGraph • LangChain • Groq • Streamlit</div>',
    unsafe_allow_html=True,
)