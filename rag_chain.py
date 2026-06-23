import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def load_vectorstore():
    print("Loading vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✓ Vector store loaded")
    return vectorstore

def build_chain(vectorstore):
    print("Building RAG chain with AWS Bedrock...")

    # 換成 Bedrock！
    llm = ChatBedrock(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
        model_kwargs={"temperature": 0.3, "max_tokens": 1000}
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    def format_docs(docs):
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content.strip()
            if len(content) > 50:
                results.append(f"[Source {i}: {source} p.{page}]\n{content}")
        return "\n\n---\n\n".join(results) if results else "No relevant content found."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AWS Solutions Architect assistant helping enterprise clients understand AWS services.

Your job:
- Answer questions clearly and concisely based ONLY on the provided context
- If the context contains relevant information, give a confident, structured answer
- If the context does NOT contain enough information, say: "I couldn't find specific information about this in the AWS documentation provided."
- Always mention which AWS service or feature you're referring to
- Use bullet points for multi-part answers
- Keep answers focused and business-relevant

Context from AWS documentation:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"])),
            source_documents=lambda x: retriever.invoke(x["input"])
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    print("✓ RAG chain ready with AWS Bedrock\n")
    return chain_with_history

def ask(chain, question, session_id="default"):
    response = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    answer = response["answer"]
    sources = response.get("source_documents", [])

    print(f"Q: {question}")
    print(f"A: {answer}\n")
    print("Sources:")
    seen = set()
    for doc in sources:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen and len(doc.page_content.strip()) > 50:
            seen.add(key)
            print(f"  • {source} — page {page}")
    print("\n" + "-" * 50 + "\n")
    return answer

if __name__ == "__main__":
    print("=== Day 5: RAG with AWS Bedrock ===\n")
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)

    questions = [
        "What is Amazon Bedrock?",
        "What is S3 versioning and why would I use it?",
    ]

    for q in questions:
        ask(chain, q)

    print("=== Day 5 完成！現在用 AWS Bedrock 驅動！===")