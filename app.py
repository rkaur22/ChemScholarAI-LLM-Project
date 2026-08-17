import os

# Must be set before torch/transformers/huggingface_hub are imported anywhere
# (including transitively, via rag.rag -> embeddings.specter2_embedder).
# Works around a macOS fork/OpenMP conflict between torch's bundled OpenMP
# runtime and huggingface_hub's multiprocessing-based parallel downloader.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

import config
from rag.rag import answer_query, submit_feedback

st.title("Computational Chemistry RAG")

user_input = st.text_input(
    "Enter a question about computational chemistry papers",
    placeholder="e.g. What are recent advances in ML interatomic potentials?",
)

if st.button("Ask") and user_input:
    with st.spinner("Processing..."):
        try:
            result = answer_query(user_input, k=config.rag.top_k)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            result = None
    st.session_state["last_result"] = result
    st.session_state["feedback_given"] = False

result = st.session_state.get("last_result")

if result:
    st.markdown("### Answer")
    st.write(result["answer"])
    st.caption(f"Answered in {result['latency_ms']} ms")

    if result["sources"]:
        st.markdown("### Sources")
        for i, paper in enumerate(result["sources"], start=1):
            title = paper.get("title", "Untitled")
            score = paper.get("score")
            label = f"[{i}] {title}" + (f"  ·  score {score:.3f}" if score is not None else "")
            with st.expander(label):
                st.write(paper.get("abstract", ""))
                if paper.get("pdf_url"):
                    st.markdown(f"[Open PDF]({paper['pdf_url']})")
    else:
        st.info("No matching papers were found for this question.")
        
    log_id = result.get("log_id")
    if log_id and not st.session_state.get("feedback_given"):
        st.markdown("Was this answer helpful?")
        col_up, col_down, _ = st.columns([1, 1, 8])
        if col_up.button("👍", key=f"up_{log_id}"):
            submit_feedback(log_id, 1)
            st.session_state["feedback_given"] = True
            st.toast("Thanks for the feedback!")
            st.rerun()
        if col_down.button("👎", key=f"down_{log_id}"):
            submit_feedback(log_id, -1)
            st.session_state["feedback_given"] = True
            st.toast("Thanks for the feedback!")
            st.rerun()
    elif st.session_state.get("feedback_given"):
        st.caption("Feedback recorded — thank you.")
