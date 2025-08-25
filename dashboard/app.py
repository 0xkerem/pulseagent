"""
PulseAgent — Streamlit Dashboard
Interactive UI to run the pipeline and visualize results.
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import nest_asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Streamlit runs inside its own event loop — nest_asyncio allows asyncio.run() inside it
nest_asyncio.apply()

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from graph.pipeline import run_pipeline
from models import PipelineState

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseAgent",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: #38bdf8; }
  .metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; }
  .priority-P0 { color: #ef4444; font-weight: 700; }
  .priority-P1 { color: #f97316; font-weight: 700; }
  .priority-P2 { color: #eab308; }
  .priority-P3 { color: #64748b; }
  .alert-critical { border-left: 3px solid #ef4444; padding-left: 10px; }
  .alert-warning { border-left: 3px solid #f97316; padding-left: 10px; }
  .alert-info { border-left: 3px solid #3b82f6; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔊 PulseAgent")
    st.caption("Product Review Intelligence")
    st.divider()

    product_name = st.text_input("Product / Company", value="Notion")
    use_fixtures = st.checkbox("Use fixture data (demo)", value=True)

    if not use_fixtures:
        st.info("Live scraping uses Reddit public API and App Store RSS.")
        subreddits_raw = st.text_input("Reddit subreddits (comma-separated)", value="Notion,productivity")
        subreddits = [s.strip() for s in subreddits_raw.split(",") if s.strip()]
        app_store_id = st.text_input("App Store App ID (optional)", value="")
    else:
        subreddits = []
        app_store_id = None

    review_limit = st.slider("Max reviews", min_value=10, max_value=200, value=50, step=10)
    docs_dir = st.text_input("Product docs folder (optional)", value="./data/docs")

    st.divider()
    run_button = st.button("▶ Run PulseAgent", type="primary", use_container_width=True)


# ── session state ─────────────────────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


# ── run pipeline ──────────────────────────────────────────────────────────────
if run_button:
    with st.spinner(f"Running PulseAgent for **{product_name}**..."):
        try:
            result: PipelineState = asyncio.run(
                run_pipeline(
                    product_name=product_name,
                    use_fixtures=use_fixtures,
                    fixture_dir="./data/fixtures",
                    docs_dir=docs_dir if Path(docs_dir).exists() else None,
                    review_limit=review_limit,
                    reddit_subreddits=subreddits,
                    app_store_app_id=app_store_id or None,
                )
            )
            st.session_state.pipeline_result = result
            st.success(f"✅ Done! Processed {len(result.raw_reviews)} reviews.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

result: PipelineState | None = st.session_state.pipeline_result

# ── main content ──────────────────────────────────────────────────────────────
if result is None:
    st.markdown("## Welcome to PulseAgent 👋")
    st.markdown(
        "Configure the product and click **▶ Run PulseAgent** in the sidebar to start. "
        "Demo fixture data for **Notion**, **Figma**, and **Linear** is included."
    )

    st.info("**How it works:** Scraper → Classifier → Urgency Scorer → RAG → "
            "Response Generator → Roadmap Builder → Trend Detector")
    st.stop()


# ── metrics row ───────────────────────────────────────────────────────────────
st.markdown(f"## 📊 Results — *{result.product_name}*")

churn_count = sum(1 for r in result.classified_reviews if r.is_churn_signal)
p0_count = sum(1 for i in result.roadmap_items if i.priority.value == "P0")

col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (col1, len(result.raw_reviews), "Reviews Analyzed"),
    (col2, len(result.clusters), "Issue Clusters"),
    (col3, churn_count, "Churn Signals 🚨"),
    (col4, len(result.roadmap_items), "Roadmap Items"),
    (col5, len(result.trend_alerts), "Trend Alerts"),
]
for col, value, label in metrics:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🗺️ Roadmap", "💬 Draft Responses", "🔥 Trends", "🔍 Raw Data"
])


# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Category Distribution")
        if result.classified_reviews:
            from collections import Counter
            cat_counts = Counter(r.category.value for r in result.classified_reviews)
            fig = px.pie(
                values=list(cat_counts.values()),
                names=list(cat_counts.keys()),
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Sentiment Distribution")
        if result.classified_reviews:
            sent_counts = Counter(r.sentiment.value for r in result.classified_reviews)
            colors = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8", "mixed": "#f97316"}
            fig2 = px.bar(
                x=list(sent_counts.keys()),
                y=list(sent_counts.values()),
                color=list(sent_counts.keys()),
                color_discrete_map=colors,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                showlegend=False,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Urgency heatmap
    st.subheader("Issue Clusters — Urgency Heatmap")
    if result.clusters:
        cluster_df = pd.DataFrame([
            {
                "Theme": c.theme[:40],
                "Category": c.category.value,
                "Reviews": c.total_count,
                "Avg Urgency": c.avg_urgency,
                "Churn Risk": c.churn_risk_count,
            }
            for c in result.clusters[:15]
        ])
        fig3 = px.scatter(
            cluster_df,
            x="Reviews",
            y="Avg Urgency",
            size="Churn Risk",
            color="Category",
            text="Theme",
            size_max=40,
        )
        fig3.update_traces(textposition="top center", textfont_size=9)
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 10]),
            height=420,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ── TAB 2: Roadmap ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Product Roadmap — AI Generated")
    if not result.roadmap_items:
        st.info("No roadmap items generated.")
    else:
        priority_colors = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "⚪"}
        effort_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}

        for item in result.roadmap_items:
            with st.expander(
                f"{priority_colors[item.priority.value]} **[{item.priority.value}]** {item.title} "
                f"— {item.category.value.replace('_', ' ').title()} "
                f"| ~{item.affected_users_estimate} users affected"
            ):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**Description:** {item.description}")
                    st.markdown(f"**User Story:** *{item.user_story}*")
                    if item.acceptance_criteria:
                        st.markdown("**Acceptance Criteria:**")
                        for ac in item.acceptance_criteria:
                            st.markdown(f"  - {ac}")
                with col_b:
                    st.markdown(f"**ID:** `{item.item_id}`")
                    st.markdown(f"**Effort:** {effort_colors.get(item.implementation_effort, '?')} {item.implementation_effort}")
                    st.markdown(f"**Churn Risk:** {item.churn_risk_score:.0f} users")
                    if item.competitor_has_it is not None:
                        st.markdown(f"**Competitor has it:** {'✅ Yes' if item.competitor_has_it else '❌ No'}")


# ── TAB 3: Draft Responses ────────────────────────────────────────────────────
with tab3:
    st.subheader("Draft Responses — Pending Human Approval")
    if not result.draft_responses:
        st.info("No draft responses generated (no reviews above urgency threshold).")
    else:
        id_map = {r.review.id: r for r in result.scored_reviews}
        for i, draft in enumerate(result.draft_responses, 1):
            scored = id_map.get(draft.review_id)
            with st.expander(
                f"Draft #{i} — Urgency: {scored.urgency_score:.1f}/10 "
                f"| {scored.category.value} "
                f"{'🚨 CHURN SIGNAL' if scored.is_churn_signal else ''}"
                if scored else f"Draft #{i}"
            ):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Original Review:**")
                    st.markdown(f"> {scored.review.text[:400] if scored else 'N/A'}")
                    if scored:
                        st.caption(f"Source: {scored.review.source} | Rating: {scored.review.rating or 'N/A'}")
                with col2:
                    st.markdown("**Draft Response:**")
                    st.markdown(draft.draft)
                    st.caption(
                        f"RAG context used: {'✅' if draft.rag_context_used else '❌'} "
                        f"| Requires approval: {'✅' if draft.requires_human_approval else '❌'}"
                    )

                col_approve, col_edit, col_reject = st.columns(3)
                col_approve.button("✅ Approve", key=f"approve_{i}")
                col_edit.button("✏️ Edit", key=f"edit_{i}")
                col_reject.button("❌ Reject", key=f"reject_{i}")


# ── TAB 4: Trends ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Trend Alerts")
    if not result.trend_alerts:
        st.info("Not enough historical data to detect trends (need reviews across multiple time windows).")
    else:
        for alert in result.trend_alerts:
            level_colors = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
            direction_arrow = "📈" if alert.direction == "rising" else "📉"
            with st.container():
                st.markdown(
                    f"{level_colors[alert.alert_level]} **{alert.theme}** "
                    f"{direction_arrow} {abs(alert.change_percent):.0f}% {alert.direction}"
                )
                st.caption(alert.summary)
                st.divider()


# ── TAB 5: Raw Data ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("Classified Reviews")
    if result.classified_reviews:
        df = pd.DataFrame([
            {
                "ID": r.review.id[:8],
                "Source": r.review.source,
                "Category": r.category.value,
                "Sentiment": r.sentiment.value,
                "Score": f"{r.sentiment_score:.2f}",
                "Churn": "🚨" if r.is_churn_signal else "",
                "Text": r.review.text[:100] + "...",
            }
            for r in result.classified_reviews
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download button
        json_data = json.dumps(
            [r.model_dump() for r in result.classified_reviews],
            indent=2,
            default=str,
        )
        st.download_button(
            "⬇️ Download classified reviews (JSON)",
            data=json_data,
            file_name=f"pulseagent_{result.product_name}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
