import streamlit as st

from src.rag.baseline import SimpleRAG
from src.rag.self_rag import SelfRAG
from src.rag.bm25_rag import BM25RAG
from src.rag.graph_rag import GraphRAG  

# Init Models
@st.cache_resource
def load_models():
    baseline = SimpleRAG()
    self_rag = SelfRAG()
    bm25 = BM25RAG()
    graph = GraphRAG() 
    return baseline, self_rag, bm25, graph


baseline_model, self_rag_model, bm25_model, graph_model = load_models()

# Page Config
st.set_page_config(page_title="RAG Hallucination Analysis", layout="wide")

st.title("📊 RAG Hallucination Analysis (Finance)")
st.markdown("Compare **Vector RAG**, **BM25 RAG**, **Graph RAG**, and **Self-RAG**")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    "This app compares multiple RAG retrieval strategies and evaluates hallucination using Self-RAG verification."
)

# Input
query = st.text_input("🔍 Enter your question:")

# Run Button
if st.button("Run Comparison"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("Running models..."):

            baseline_result = baseline_model.query(query)
            self_result = self_rag_model.self_rag_query(query)
            bm25_result = bm25_model.query(query)
            graph_result = graph_model.query(query)  

        st.divider()

        # Output
        col1, col2, col3, col4 = st.columns(4)

        # Vector RAG
        with col1:
            st.markdown("## 🧠 Vector RAG")
            st.write(baseline_result["answer"])

            for i, doc in enumerate(baseline_result["context"]):
                with st.expander(f"Doc {i+1}"):
                    st.write(doc["content"])

        # Self RAG
        with col2:
            st.markdown("## 🔍 Self-RAG")
            st.write(self_result["answer"])

            for i, doc in enumerate(self_result["context"]):
                with st.expander(f"Doc {i+1}"):
                    st.write(doc["content"])

        # BM25 RAG
        with col3:
            st.markdown("## 📚 BM25 RAG")
            st.write(bm25_result["answer"])

            for i, doc in enumerate(bm25_result["context"]):
                with st.expander(f"Doc {i+1}"):
                    st.write(doc["content"])

        # Graph RAG
        with col4:
            st.markdown("## 🕸️ Graph RAG")
            st.write(graph_result["answer"])

            for i, doc in enumerate(graph_result["context"]):
                with st.expander(f"Doc {i+1}"):
                    st.write(doc["content"])

        st.divider()

        # Self RAG Verification
        st.subheader("📌 Self-RAG Verification Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Relevant", str(self_result.get("is_relevant")))
        m2.metric("Grounded", str(self_result.get("is_grounded")))
        m3.metric("Useful", str(self_result.get("is_useful")))
        m4.metric("Retries", self_result.get("retries", 0))

        st.divider()

        # Hallucination Analysis
        st.subheader("🚨 Hallucination Analysis")

        if self_result.get("is_grounded") is False:
            st.error("❌ Self-RAG detected hallucination (answer not grounded in context).")
        else:
            st.success("✅ Answer is grounded in retrieved documents.")

        st.divider()

        # Comparison Insights
        st.subheader("📊 Comparison Insight")

        # BM25 vs Vector
        if bm25_result["answer"].strip() != baseline_result["answer"].strip():
            st.info("📌 BM25 vs Vector RAG produce different answers → retrieval method impacts results.")

        # Graph insight
        if graph_result["answer"].strip() != baseline_result["answer"].strip():
            st.info("🕸️ Graph RAG produces different results (strong for entity-based queries).")

        if len(graph_result["context"]) == 0:
            st.warning("⚠️ Graph RAG retrieved no documents (likely no entities in query).")

        # Self-RAG correction
        if self_result.get("is_grounded") is False:
            st.warning("⚠️ Self-RAG flagged the generated answer as unreliable.")

        # Answer difference
        if baseline_result["answer"].strip() != self_result["answer"].strip():
            st.info("🔍 Self-RAG modified the answer compared to baseline.")

        # Safe behavior
        if (
            self_result.get("is_grounded") is True
            and self_result.get("is_relevant") is True
        ):
            st.success("📈 Self-RAG confirms the answer is reliable and relevant.")