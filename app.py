import streamlit as st
import os
import re
import sys
import io
import contextlib
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.tools import tool
import requests
from bs4 import BeautifulSoup

# --- Page Config ---
st.set_page_config(
    page_title="Company Researcher AI Portal",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');
    h1, h2, h3, .main-title { font-family: 'Outfit', sans-serif !important; }
    body, p, span, div, label { font-family: 'Inter', sans-serif !important; }

    .hero-container {
        padding: 2rem 2.5rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(99,102,241,0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99,102,241,0.4) !important;
        opacity: 0.95;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.15);
    }
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 8px !important;
        border: 1px solid rgba(128,128,128,0.15) !important;
        background-color: rgba(21,27,44,0.6) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Groq cache_breakpoint fix ---
# CrewAI enables prompt caching via `cache_control` / `cache_breakpoint` fields
# injected into messages. Groq rejects these fields entirely.
# Fix: (1) turn off litellm caching globally, (2) patch every completion entry
# point to scrub the fields before they reach the HTTP layer.
import litellm as _litellm

# Disable litellm-level prompt caching globally
_litellm.cache = None
_litellm.enable_cache = False

def _scrub_messages(messages):
    """Remove cache_breakpoint / cache_control from every message and content block."""
    if not isinstance(messages, list):
        return messages
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg.pop("cache_breakpoint", None)
        msg.pop("cache_control", None)
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_breakpoint", None)
                    block.pop("cache_control", None)
    return messages

def _make_patched(original):
    def _patched(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _scrub_messages(kwargs["messages"])
        elif len(args) > 1:
            args = (args[0], _scrub_messages(args[1])) + args[2:]
        kwargs.pop("cache_control", None)
        return original(*args, **kwargs)
    return _patched

# Patch all litellm completion entry points
_litellm.completion = _make_patched(_litellm.completion)
try:
    import litellm.main as _lm
    _lm.completion = _litellm.completion
except Exception:
    pass
try:
    from litellm import Router as _Router
    _orig_router_completion = _Router.completion
    def _patched_router_completion(self, *args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _scrub_messages(kwargs["messages"])
        elif len(args) > 1:
            args = (args[0], _scrub_messages(args[1])) + args[2:]
        return _orig_router_completion(self, *args, **kwargs)
    _Router.completion = _patched_router_completion
except Exception:
    pass

# --- Custom Scraper Tool ---
@tool
def scraper_tool(url: str):
    """Useful to scrape content from a website."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000]
    except Exception as e:
        return f"Error scraping {url}: {e}"

def capture_crew_output(crew, inputs):
    """
    Runs crew.kickoff() while silently capturing ALL stdout and stderr
    into a string buffer. No st.empty() or placeholder updates during
    execution — this completely eliminates the text bleeding issue.
    """
    buffer = io.StringIO()
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = buffer
    sys.stderr = buffer
    
    result = ""
    error_message = ""
    success = False
    
    try:
        result_object = crew.kickoff(inputs=inputs)
        result = str(result_object)
        success = True
    except Exception as e:
        error_message = str(e)
        success = False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    raw_log = buffer.getvalue()
    clean_log = ansi_escape.sub('', raw_log)
    
    return success, result, error_message, clean_log

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/search-in-list.png", width=80)
    st.markdown("### ⚙️ Settings & Credentials")
    st.markdown("Configure your AI models and API credentials below.")
    
    st.markdown("---")
    st.markdown("#### 🤖 LLM Provider Settings")
    llm_provider = st.selectbox(
        "Select Provider",
        ["Gemini (Google)", "OpenAI", "Anthropic (Claude)", "Groq"],
        index=0
    )
    
    if llm_provider == "Gemini (Google)":
        model_choice = st.selectbox(
            "Select Model",
            ["gemini/gemini-2.0-flash", "gemini/gemini-2.0-flash-lite", "gemini/gemini-2.5-flash"]
        )
        llm_doc_url = "https://aistudio.google.com/"
        env_key_name = "GEMINI_API_KEY"
    elif llm_provider == "OpenAI":
        model_choice = st.selectbox(
            "Select Model",
            ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4-turbo"]
        )
        llm_doc_url = "https://platform.openai.com/api-keys"
        env_key_name = "OPENAI_API_KEY"
    elif llm_provider == "Anthropic (Claude)":
        model_choice = st.selectbox(
            "Select Model",
            ["anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-haiku-20240307"]
        )
        llm_doc_url = "https://console.anthropic.com/settings/keys"
        env_key_name = "ANTHROPIC_API_KEY"
    else:  # Groq
        model_choice = st.selectbox(
            "Select Model",
            [
                "groq/llama-3.3-70b-versatile",
                "groq/llama-3.1-8b-instant",
                "groq/mixtral-8x7b-32768",
                "groq/gemma2-9b-it",
            ]
        )
        llm_doc_url = "https://console.groq.com/keys"
        env_key_name = "GROQ_API_KEY"

    default_llm_key = os.environ.get(env_key_name, "")
    llm_api_key = st.text_input(
        f"{llm_provider} API Key",
        value=default_llm_key,
        type="password",
        help=f"Get your API key from [here]({llm_doc_url})."
    )
    
    st.markdown("---")
    st.markdown("#### 🔍 Search Tool Settings")
    default_serper_key = os.environ.get("SERPER_API_KEY", "")
    serper_api_key = st.text_input(
        "Serper API Key",
        value=default_serper_key,
        type="password",
        help="Required for web search. Get a free key at [Serper.dev](https://serper.dev)."
    )
    
    st.markdown("---")
    st.markdown("#### 🔋 Status Checks")
    if llm_api_key:
        st.markdown("🟢 **LLM Provider API Key:** Configured")
    else:
        st.markdown(f"🔴 **LLM Provider API Key:** Missing")
    if serper_api_key:
        st.markdown("🟢 **Serper API Key:** Configured")
    else:
        st.markdown("🔴 **Serper API Key:** Missing")

# --- Main Layout ---
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🔍 Company Researcher AI</h1>
    <p class="hero-subtitle">Harness multi-agent systems to conduct comprehensive, real-time background, product, and news research on any company or product.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 🎯 Start Your Search")
    company = st.text_input(
        "Enter Company or Product Name",
        placeholder="e.g. Stripe, OpenAI, NVIDIA...",
        label_visibility="collapsed"
    )
with col2:
    st.write("")
    st.write("")
    start_button = st.button("🚀 Conduct Research", use_container_width=True)

# --- Execution ---
if start_button:
    if not company:
        st.error("⚠️ Please enter a company or product name.")
    elif not llm_api_key:
        st.error(f"⚠️ Missing {llm_provider} API Key in the sidebar.")
    elif not serper_api_key:
        st.error("⚠️ Missing Serper API Key in the sidebar.")
    else:
        # Set env vars — remove GOOGLE_API_KEY to avoid conflict warning
        os.environ["SERPER_API_KEY"] = serper_api_key
        os.environ[env_key_name] = llm_api_key
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
        # Groq: LiteLLM also reads GROQ_API_KEY from the environment
        if llm_provider == "Groq":
            os.environ["GROQ_API_KEY"] = llm_api_key

        # --- Build crew ---
        crew = None
        init_error = ""
        try:
            llm = LLM(model=model_choice, api_key=llm_api_key, temperature=0.2)
            search_tool = SerperDevTool()

            researcher = Agent(
                role='Web Researcher',
                goal='Find recent and relevant information about {company}.',
                backstory='You are an expert researcher with a knack for finding the latest news and information.',
                tools=[search_tool],
                llm=llm
            )
            extractor = Agent(
                role='Data Extractor',
                goal='Scrape and summarize information about {company}.',
                backstory='You are excellent at parsing raw website content into clean, structured reports.',
                tools=[scraper_tool],
                llm=llm
            )
            search_task = Task(
                description='Search the web for news, overview, and products for {company}.',
                expected_output='A summary of findings from web searches.',
                agent=researcher
            )
            report_task = Task(
                description='Based on the research, compile a structured report for {company}.',
                expected_output='A structured report covering background, products, news, and market position.',
                agent=extractor,
                context=[search_task]
            )
            crew = Crew(
                agents=[researcher, extractor],
                tasks=[search_task, report_task],
                process=Process.sequential,
                verbose=False
            )
        except Exception as e:
            init_error = str(e)

        if init_error:
            st.error(f"❌ Failed to initialize crew: {init_error}")
        else:
            # --- Run crew with spinner — NO st.empty() or placeholder updates ---
            with st.spinner("🤖 Agents are researching... please wait"):
                success, result, error_message, log_text = capture_crew_output(
                    crew, inputs={'company': company}
                )

            # --- Display results cleanly AFTER crew finishes ---
            if success:
                st.success("✨ Research Completed!")
                tab1, tab2 = st.tabs(["📊 Research Report", "⚙️ Agent Log"])

                with tab1:
                    st.markdown("""
                    <div style="padding:1.5rem;background:rgba(99,102,241,0.05);border-radius:8px;border:1px solid rgba(99,102,241,0.2);margin-bottom:1rem;">
                        <h4 style="margin:0;color:#6366f1;">📑 Executive Dossier</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(result)
                    st.download_button(
                        label="📥 Download Report (.md)",
                        data=result,
                        file_name=f"{company.lower().replace(' ', '_')}_report.md",
                        mime="text/markdown"
                    )

                with tab2:
                    st.subheader("Agent Thought Logs")
                    st.code(log_text, language="text")

            else:
                st.error(f"❌ An error occurred during execution: {error_message}")
                if log_text:
                    st.subheader("Execution Log")
                    st.code(log_text, language="text")