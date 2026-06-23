import os
import streamlit as st
from rag_chain import load_vectorstore, build_chain

st.set_page_config(
    page_title="AWS Docs Assistant",
    page_icon="☁️",
    layout="wide"
)

@st.cache_resource
def init_chain():
    if not os.path.exists("vectorstore"):
        st.info("Building knowledge base for the first time... this takes ~3 minutes.")
        os.system("python scripts/download_docs.py")
        os.system("python scripts/ingest.py")
    vectorstore = load_vectorstore()
    return build_chain(vectorstore)

chain = init_chain()

with st.sidebar:
    st.title("☁️ AWS Docs Assistant")
    st.markdown("Ask anything about **Amazon Bedrock** or **Amazon S3** based on official AWS documentation.")
    st.divider()
    st.markdown("**Knowledge Base**")
    st.markdown("- 📄 Amazon Bedrock User Guide")
    st.markdown("- 📄 Amazon S3 User Guide")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = "session_" + str(st.session_state.get("clear_count", 0) + 1)
        st.session_state.clear_count = st.session_state.get("clear_count", 0) + 1
        st.rerun()
    st.divider()
    st.caption("Built with LangChain + FAISS + AWS Bedrock")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "session_0"

st.title("AWS Documentation Chatbot")
st.caption("Powered by RAG — answers are grounded in official AWS docs")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for source in msg["sources"]:
                    st.caption(f"• {source}")

if prompt := st.chat_input("Ask about AWS services..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching AWS docs..."):
            response = chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )
            answer = response["answer"]
            sources = response.get("source_documents", [])

            seen = set()
            source_list = []
            for doc in sources:
                source = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page", "?")
                key = f"{source}_p{page}"
                if key not in seen and len(doc.page_content.strip()) > 50:
                    seen.add(key)
                    source_list.append(f"{source} — page {page}")

        st.markdown(answer)
        if source_list:
            with st.expander("📚 Sources"):
                for source in source_list:
                    st.caption(f"• {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_list
    })
