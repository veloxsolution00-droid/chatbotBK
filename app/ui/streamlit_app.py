import os, requests, streamlit as st
API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Offline Document Chatbot", page_icon="📚", layout="wide")
st.title("Offline Document Intelligence Chatbot")
st.caption("Local-only document chat. No account setup is required. Keep the backend bound to 127.0.0.1 for single-computer use.")
if "messages" not in st.session_state: st.session_state.messages = []

def api_get(path, timeout=10):
    r = requests.get(API + path, timeout=timeout); r.raise_for_status(); return r.json()
def api_post(path, **kwargs):
    r = requests.post(API + path, **kwargs); r.raise_for_status(); return r.json()
def citation_line(sources):
    return " · ".join(f"[{source.get('rank', index)}] {source.get('filename', 'document')} — {source.get('source_location', 'location unavailable')}"
                    for index, source in enumerate(sources, 1))

with st.sidebar:
    st.subheader("System status")
    health_limits = {}
    try:
        h = requests.get(API + "/health", timeout=5).json()
        health_limits = h.get("limits") or {}
        st.write("Backend:", h.get("status"))
        ollama = h.get("ollama", {})
        st.write("Ollama:", "Connected" if ollama.get("available") else "Unavailable")
        st.write("Generation model:", ollama.get("generation_model"), "✓" if ollama.get("generation_model_available") else "missing")
        st.write("Embedding model:", ollama.get("embedding_model"), "✓" if ollama.get("embedding_model_available") else "missing")
        st.write("ChromaDB:", h.get("chromadb", {}).get("available"), "· chunks:", h.get("chromadb", {}).get("count", 0))
        st.caption(f"Per file: {h.get('limits',{}).get('max_file_mb',100)} MB · Batch: {h.get('limits',{}).get('max_batch_mb',500)} MB")
    except Exception as e: st.warning(f"Backend not reachable. Start FastAPI first. {e}")

    st.subheader("Upload documents")
    if "uploader_id" not in st.session_state:
        st.session_state.uploader_id = 0
    up = st.file_uploader("Select multiple documents", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_id}")
    if st.button("Upload and process", disabled=not up, use_container_width=True):
        try:
            total = sum(f.size for f in up)
            max_batch_mb = int(health_limits.get("max_batch_mb", 500) or 500)
            if total > max_batch_mb * 1024 * 1024: st.error(f"Batch exceeds {max_batch_mb} MB.")
            else:
                files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in up]
                with st.spinner("Saving files and adding ingestion jobs to the local queue..."):
                    data = api_post("/documents/upload", files=files, timeout=600)
                st.success(f"Accepted {data['files']} file(s), {data['total_bytes'] / (1024*1024):.1f} MB. Staging: {data['staging_seconds']} sec.")
                st.info("The job list updates automatically with the current extraction, chunking, embedding, and indexing stage.")
                st.session_state.uploader_id += 1
                st.rerun()
        except Exception as e: st.error(f"Upload failed: {e}")

@st.fragment(run_every="3s")
def show_jobs_and_documents():
    st.subheader("Ingestion jobs")
    try:
        job_data = api_get("/jobs", timeout=10).get("jobs", [])
        active = [j for j in job_data if j.get("status") in ("queued", "processing")]
        if active: st.caption(f"Auto-refreshing every 3 seconds · {len(active)} job(s) active")
        if not job_data: st.info("No ingestion jobs yet. Upload a document from the sidebar.")
        for j in job_data[:50]:
            label = f"{j['filename']} · {j['status']} · {j.get('progress',0)}%"
            with st.expander(label, expanded=j["status"] in ("queued", "processing", "failed")):
                st.progress(max(0,min(100,int(j.get("progress",0))))/100)
                st.write("Stage:", j.get("stage") or j["status"])
                st.write("Updated:", j.get("updated_at", "—"))
                st.write("Size:", f"{j['size_bytes']/(1024*1024):.1f} MB")
                if j.get("document_id"): st.write("Document ID:", j["document_id"])
                if j.get("error"): st.error(j["error"])
                if j["status"] == "failed" and st.button("Retry", key="retry_"+j["job_id"]):
                    try: api_post("/jobs/"+j["job_id"]+"/retry", timeout=10); st.rerun()
                    except Exception as e: st.error(str(e))
    except Exception as e: st.info(f"Jobs are not available: {e}")

    st.subheader("Your documents")
    try:
        docs = api_get("/documents", timeout=10).get("documents", [])
        if not docs: st.info("No documents have been indexed yet.")
        for d in docs:
            a,b,c = st.columns([5,2,1]); a.write(d["filename"])
            b.write(f"{d.get('processing_status','unknown')} · {d.get('chunk_count',0)} chunks")
            if d.get("processing_error"): a.caption("Issue: " + d["processing_error"])
            if c.button("Delete", key="delete_"+d["document_id"]):
                r = requests.delete(API+"/documents/"+d["document_id"], timeout=30); r.raise_for_status(); st.rerun()
    except Exception as e: st.info(f"Document inventory unavailable: {e}")

show_jobs_and_documents()

st.subheader("Ask your documents")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + citation_line(m["sources"]))
        for src in m.get("sources", []):
            with st.expander(f"{src.get('filename')} — {src.get('source_location')}"):
                st.write("Relevance:", src.get("relevance_score", "—"))
                st.write(src.get("supporting_text", ""))
q = st.chat_input("Ask a question about your uploaded documents")
if q:
    st.session_state.messages.append({"role":"user", "content":q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Resolving context, retrieving supporting passages, and generating a local answer..."):
                # The API caps each history message at 2000 characters. Answers can
                # be longer, so trim here instead of failing the next request with 422.
                history = [{"role": m["role"], "content": str(m["content"])[:2000]} for m in st.session_state.messages[:-1][-8:] if m.get("role") in ("user", "assistant")]
                data = api_post("/chat", json={"question":q, "history":history}, timeout=600)
            st.markdown(data.get("answer", "No answer returned."))
            retrieval = data.get("retrieval", {})
            if data.get("question_resolved"):
                st.caption("Interpreted follow-up as: " + data["question_resolved"])
            if data.get("question_type"):
                st.caption("Question type: " + data["question_type"])
            st.caption(f"Status: {data.get('status','unknown')} · Response time: {data.get('response_seconds', '—')} seconds")
            confidence = data.get("confidence") or {}
            if confidence.get("label"):
                score = confidence.get("score")
                label = confidence["label"]
                heading = f"Confidence: {label}" + (f" ({score:.0%})" if isinstance(score, (int, float)) else "")
                (st.success if label == "high" else st.info if label == "medium" else st.warning)(heading)
                signals = confidence.get("signals") or {}
                if signals:
                    st.caption(f"Signals: retrieval similarity {signals.get('retrieval_similarity', '—')} · term coverage {signals.get('term_coverage', '—')} · citation coverage {signals.get('citation_coverage', '—')} · verifier {signals.get('verified', '—')}")
            if retrieval:
                st.caption(f"Retrieval diagnostics: {retrieval.get('selected_chunks', 0)} chunk(s) selected / {retrieval.get('candidate_chunks', 0)} candidate(s) · best similarity: {retrieval.get('best_similarity', '—')} · threshold: {retrieval.get('min_similarity', '—')} · pass(es): {retrieval.get('retrieval_passes', 1)}")
            citation_check = data.get("citation_validation")
            # An abstention also carries an invalid citation block (the refusal has
            # no [n] markers). Reporting that as a citation failure contradicts the
            # real status, which is "no answer in the evidence".
            if data.get("status") == "citation_validation_failed" and citation_check and not citation_check.get("valid", False):
                st.warning("Citation validation did not pass. The response may have been withheld because its source reference was missing or invalid.")
            verification = data.get("verification") or {}
            if verification.get("checked") and not verification.get("passed"):
                st.warning("The local verifier could not confirm this answer against the cited passages. Treat it as unverified and check the sources below.")
            conflicts = data.get("potential_conflicts") or []
            if conflicts:
                st.warning(f"Potential evidence disagreement detected in {len(conflicts)} passage pair(s). Review the cited passages; this is a heuristic warning, not a confirmed contradiction.")
            if data.get("error"): st.error(data["error"])
            if data.get("sources"):
                st.caption("Sources: " + citation_line(data["sources"]))
            for src in data.get("sources", []):
                with st.expander(f"{src.get('filename')} — {src.get('source_location')}"):
                    st.write("Relevance:", src.get("relevance_score", "—"))
                    st.write(src.get("supporting_text", ""))
            st.session_state.messages.append({"role":"assistant", "content":data.get("answer", "No answer returned."), "sources":data.get("sources",[])})
        except Exception as e:
            st.error(f"Chat request failed: {e}")
            st.session_state.messages.append({"role":"assistant", "content":str(e)})
