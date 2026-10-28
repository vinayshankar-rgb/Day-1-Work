import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

### Hide the annoying Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


### --- 1. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Professor Vector", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Professor Vector's Open-Book Exam Cheat Bot")
st.subheader("Day 1 Workshop Deliverable")

### Sidebar for safe API Key collection
st.sidebar.header("Configuration")
user_api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")


### --- 2. KNOWLEDGE BASE DATA ---
@st.cache_resource
def load_knowledge_base():
    documents = [
        "Advanced DBMS: ACID properties guarantee that database transactions are processed reliably.",
        "Advanced DBMS: A Vector Database stores high-dimensional embeddings for semantic search mapping.",
        "Computer Organization (COA): Amdahl's law computes the theoretical speedup in latency of execution.",
        "Computer Organization (COA): Cache memory minimizes the average time to access data from the Main Memory.",
        "Campus Policy: The official daily student leave policy requires applying 24 hours in advance via the portal.",
        "Campus Policy: The institutional code of conduct mandates carrying physical ID cards at all times."
    ]
    # Initialize embedding model once and cache it
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(documents)
    return documents, model, doc_embeddings

# Properly load cached components
documents, embed_model, doc_embeddings = load_knowledge_base()


### --- 3. RAG CORE LOGIC FUNCTION ---

def run_rag_pipeline(query, api_key):
    import re

    # Encode the user's question
    query_embedding = embed_model.encode([query])
    similarities = embed_model.similarity(
        query_embedding, doc_embeddings
    )[0].tolist()

    # Retrieve the top 2 relevant documents
    scored_docs = sorted(
        zip(documents, similarities),
        key=lambda item: item[1],
        reverse=True,
    )

    top_chunks = [doc for doc, score in scored_docs[:2]]
    context_str = "\n".join(top_chunks)

    system_instruction = (
        "You are a helpful campus academic assistant. Answer using ONLY "
        "the provided context. If the answer is absent, reply exactly: "
        "'I do not have the relevant information in the provided document context.'"
    )

    user_prompt = (
        f"Context from documents:\n{context_str}\n\n"
        f"Question: {query}"
    )

    try:
        client = genai.Client(api_key=api_key)

        # Discover models available to this API key
        available_models = list(client.models.list())
        candidates = []

        excluded_terms = (
            "embedding", "image", "tts", "live",
            "audio", "robotics", "computer-use"
        )

        for item in available_models:
            name = getattr(item, "name", "") or ""
            actions = getattr(item, "supported_actions", []) or []
            model_id = name.removeprefix("models/")

            if (
                "generateContent" in actions
                and "flash" in model_id.lower()
                and not any(
                    term in model_id.lower()
                    for term in excluded_terms
                )
            ):
                candidates.append(model_id)

        if not candidates:
            return (
                "No compatible text-generation Flash model was "
                "found for this API key. Check your Gemini API "
                "access and available models.",
                top_chunks,
            )

        # Prefer the highest numbered version among discovered models
        def version_key(model_id):
            match = re.search(r"gemini-(\d+(?:\.\d+)*)", model_id)
            if match:
                return tuple(
                    int(part) for part in match.group(1).split(".")
                )
            return (0,)

        candidates.sort(key=version_key, reverse=True)

        # Try available models until one succeeds
        errors = []

        for model_id in candidates:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    ),
                )

                if response.text:
                    return response.text, top_chunks

                errors.append(f"{model_id}: empty response")

            except Exception as model_error:
                errors.append(f"{model_id}: {model_error}")

        return (
            "All discovered models failed. Check API access, "
            "quota, and the errors below:\n" + "\n".join(errors),
            top_chunks,
        )

    except Exception as e:
        return f"❌ API Error: {e}", []


### --- 4. APP INTERACTION LOGIC ---
query_input = st.text_input("Ask a question about DBMS, COA, or Campus Policies:")

if st.button("Run RAG Engine"):
    if not user_api_key:
        st.warning("⚠️ Please input a valid Gemini API Key in the sidebar first!")
    elif not query_input.strip():
        st.warning("⚠️ Please type a question before submitting.")
    else:
        with st.spinner("Retrieving facts and generating answer..."):
            answer, retrieved_contexts = run_rag_pipeline(query_input, user_api_key)

        # Output formatting
        if "❌ API Error" in answer:
            st.error(answer)
        else:
            st.success("### Response:")
            st.write(answer)

            if retrieved_contexts:
                with st.expander("👁️ View Retrieved Context Chunks (Open-Book Verification)"):
                    for idx, chunk in enumerate(retrieved_contexts, 1):
                        st.info(f"**Chunk {idx}:** {chunk}")
