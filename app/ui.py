import streamlit as st
from search import hybrid_search, format_chunks_for_llm
from llm import chat

# Page config
st.set_page_config(
    page_title="Local Business Review Summarizer",
    page_icon="🍽️",
    layout="wide"
)

# Header
st.title("🍽️ Local Business Review Summarizer")
st.caption("Fetches real customer reviews and delivers a structured summary of pros and cons")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Type a business name or location
    2. Reviews are fetched from the database
    3. AI analyzes and summarizes pros & cons
    4. Ask follow-up questions anytime
    """)

    st.divider()

    st.header("Try asking...")
    st.markdown("""
    - *Pros and cons of Papa Murphy's in Tucson*
    - *What are people saying about Burger Up?*
    - *Is Dio Modern Mediterranean worth visiting?*
    - *Best restaurants in Nashville?*
    - *What are the cons of Roma Pizza?*
    """)

    st.divider()

    # Show retrieved sources toggle
    show_sources = st.toggle("Show retrieved reviews", value=False)

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about a local business..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Search
        with st.spinner("Fetching relevant reviews..."):
            chunks  = hybrid_search(prompt)
            context = format_chunks_for_llm(chunks)

        # Show sources if toggle is on
        if show_sources and chunks:
            with st.expander(f"📚 Retrieved {len(chunks)} review chunks"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**{i}. {c['business_name']}** ({c['city']}, {c['state']}) — {c['review_stars']}⭐")
                    st.caption(c['text'])
                    st.divider()

        # Generate summary
        with st.spinner("Analyzing pros and cons..."):
            response = chat(prompt, context, st.session_state.history)

        st.markdown(response)

    # Save to session
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append({"role": "user",      "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": response})
