import gradio as gr
import httpx
import json

API_BASE = "http://localhost:8000"

STRATEGY_LABELS = {
    "bm25":     "BM25",
    "dense":    "Dense (BGE-large)",
    "rrf":      "RRF Hybrid",
    "reranker": "Reranker (MiniLM)",
}

STRATEGY_METRICS = {
    "bm25":     {"NDCG@10": "0.2141", "MRR@10": "0.1705", "Recall@10": "0.3613"},
    "dense":    {"NDCG@10": "0.7118", "MRR@10": "0.6632", "Recall@10": "0.8771"},
    "rrf":      {"NDCG@10": "0.5933", "MRR@10": "0.5266", "Recall@10": "0.8190"},
    "reranker": {"NDCG@10": "0.5561", "MRR@10": "0.4824", "Recall@10": "0.8030"},
}


def format_results_html(strategy_key: str, data: dict) -> str:
    """Render one strategy column as HTML."""
    label   = STRATEGY_LABELS[strategy_key]
    latency = data["latency_ms"]
    results = data["results"]
    bench   = STRATEGY_METRICS[strategy_key]

    cards = ""
    for i, r in enumerate(results, 1):
        score_pct = min(max(r["score"], 0), 1) * 100  # only meaningful for dense/rrf
        text_preview = r["text"][:280] + ("…" if len(r["text"]) > 280 else "")
        cards += f"""
        <div class="card">
            <div class="card-rank">#{i}</div>
            <div class="card-pid">PID: {r['pid']}</div>
            <div class="card-score">Score: {r['score']:.4f}</div>
            <div class="card-text">{text_preview}</div>
        </div>
        """

    metrics_html = "".join(
        f'<span class="metric"><b>{k}</b> {v}</span>'
        for k, v in bench.items()
    )

    return f"""
    <div class="col-header">
        <div class="col-title">{label}</div>
        <div class="col-latency">⏱ {latency:.1f} ms</div>
        <div class="col-metrics">{metrics_html}</div>
    </div>
    <div class="col-results">{cards}</div>
    """


def fetch_metrics_html() -> str:
    try:
        r = httpx.get(f"{API_BASE}/metrics", timeout=5)
        data = r.json()
    except Exception as e:
        return f"<p style='color:#ef4444'>Could not reach /metrics: {e}</p>"

    rows = ""
    for strategy, vals in data.items():
        label = STRATEGY_LABELS.get(strategy, strategy)
        if vals["n"] == 0:
            rows += f"<tr><td>{label}</td><td colspan='3' style='color:#94a3b8'>no data yet</td></tr>"
        else:
            rows += f"""
            <tr>
                <td>{label}</td>
                <td>{vals['p50']} ms</td>
                <td>{vals['p95']} ms</td>
                <td>{vals['n']}</td>
            </tr>"""

    return f"""
    <table class="metrics-table">
        <thead><tr>
            <th>Strategy</th><th>p50</th><th>p95</th><th>Requests (last 100)</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


def run_query(query: str):
    if not query.strip():
        empty = "<p style='color:#94a3b8;padding:1rem'>Enter a query above.</p>"
        return empty, empty, empty, empty, ""

    try:
        resp = httpx.post(
            f"{API_BASE}/query",
            json={"query": query, "top_k": 5},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        err = "<p style='color:#ef4444'>Cannot connect to backend. Is the FastAPI server running?</p>"
        return err, err, err, err, ""
    except Exception as e:
        err = f"<p style='color:#ef4444'>Error: {e}</p>"
        return err, err, err, err, ""

    bm25_html     = format_results_html("bm25",     data["bm25"])
    dense_html    = format_results_html("dense",    data["dense"])
    rrf_html      = format_results_html("rrf",      data["rrf"])
    reranker_html = format_results_html("reranker", data["reranker"])
    metrics_html  = fetch_metrics_html()

    return bm25_html, dense_html, rrf_html, reranker_html, metrics_html


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0a0a0f !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.header {
    padding: 2rem 2rem 0.5rem;
    border-bottom: 1px solid #1e293b;
}

.header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: #f8fafc;
    letter-spacing: -0.5px;
}

.header .subtitle {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}

.query-row {
    display: flex;
    gap: 0.75rem;
    padding: 1.25rem 2rem;
    align-items: flex-end;
    border-bottom: 1px solid #1e293b;
}

.gr-textbox textarea {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 0.9rem !important;
}

.gr-textbox textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

button.primary {
    background: #2563eb !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    white-space: nowrap;
}

button.primary:hover { background: #1d4ed8 !important; }

.results-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border-bottom: 1px solid #1e293b;
}

.col-header {
    padding: 0.85rem 1rem 0.6rem;
    border-bottom: 2px solid #1e293b;
    background: #0d1117;
}

.col-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #93c5fd;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.col-latency {
    font-size: 0.72rem;
    color: #4ade80;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}

.col-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 0.6rem;
    margin-top: 0.4rem;
}

.metric {
    font-size: 0.68rem;
    color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace;
}

.metric b { color: #cbd5e1; }

.col-results {
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    border-right: 1px solid #1e293b;
}

.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 0.7rem 0.85rem;
    transition: border-color 0.15s;
}

.card:hover { border-color: #3b82f6; }

.card-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3b82f6;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.card-pid {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    margin-bottom: 0.15rem;
}

.card-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #f59e0b;
    margin-bottom: 0.4rem;
}

.card-text {
    font-size: 0.78rem;
    color: #94a3b8;
    line-height: 1.5;
}

.metrics-section {
    padding: 1.25rem 2rem;
}

.metrics-section h3 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
}

.metrics-table th {
    text-align: left;
    padding: 0.4rem 0.75rem;
    color: #475569;
    border-bottom: 1px solid #1e293b;
    font-weight: 400;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metrics-table td {
    padding: 0.45rem 0.75rem;
    color: #cbd5e1;
    border-bottom: 1px solid #111827;
}

.metrics-table tr:hover td { background: #111827; }

/* hide gradio chrome */
footer { display: none !important; }
.gr-panel { background: transparent !important; border: none !important; }
"""


# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="RankBench") as demo:

    gr.HTML("""
    <div class="header">
        <h1>RankBench</h1>
        <div class="subtitle">MS MARCO Passage Retrieval · BM25 · Dense · RRF · Reranker</div>
    </div>
    """)

    with gr.Row(elem_classes="query-row"):
        query_box = gr.Textbox(
            placeholder="e.g. what causes thunder?",
            label="",
            scale=5,
            lines=1,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    # 4-column results — each column is a gr.HTML inside a row
    with gr.Row(elem_classes="results-grid"):
        bm25_out     = gr.HTML(label="BM25")
        dense_out    = gr.HTML(label="Dense")
        rrf_out      = gr.HTML(label="RRF")
        reranker_out = gr.HTML(label="Reranker")

    gr.HTML('<div class="metrics-section"><h3>Latency Metrics (last 100 requests)</h3>')
    metrics_out = gr.HTML()
    gr.HTML('</div>')

    # wire up
    search_btn.click(
        fn=run_query,
        inputs=query_box,
        outputs=[bm25_out, dense_out, rrf_out, reranker_out, metrics_out],
    )
    query_box.submit(
        fn=run_query,
        inputs=query_box,
        outputs=[bm25_out, dense_out, rrf_out, reranker_out, metrics_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
