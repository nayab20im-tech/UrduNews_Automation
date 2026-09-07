"""
monitoring/app_streamlit.py
===========================
Optional Streamlit frontend for Section 7 (the architecture diagram
suggests "Streamlit / FastAPI"). Uses the shared
:class:`monitoring.dashboard_data.DashboardData` aggregation layer, so
both frontends display identical data.

Streamlit is *not* a project dependency (the pipeline must stay fully
offline); install it separately to use this view:

    pip install streamlit
    streamlit run monitoring/app_streamlit.py
    # or: python -m monitoring.run_dashboard --backend streamlit

When Streamlit is missing, ``python -m monitoring.run_dashboard``
falls back to the zero-dependency stdlib server automatically.
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError:  # pragma: no cover - interactive guard
    raise SystemExit(
        "Streamlit is not installed. Install it with 'pip install "
        "streamlit', or run the zero-dependency dashboard instead:\n"
        "    python -m monitoring.run_dashboard"
    )

from monitoring.dashboard_data import DashboardData


def main() -> None:
    st.set_page_config(page_title="UrduNewsAI Dashboard", layout="wide")
    st.title("UrduNewsAI — Operations Dashboard")

    with DashboardData() as data:
        st.subheader("Pipeline status")
        status = data.pipeline_status()
        raw, proc, dist = status.get("raw", {}), status.get("processed", {}), \
            status.get("distribution", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw articles", raw.get("total_articles", 0))
        c2.metric("Processed", proc.get("articles", 0),
                  f"{proc.get('translated', 0)} translated")
        c3.metric("TTS audio", proc.get("with_tts_audio", 0))
        c4.metric("Published", dist.get("published", 0))
        runs = status.get("recent_runs") or []
        if runs:
            st.dataframe(runs, use_container_width=True)

        st.subheader("Latest news preview")
        st.dataframe(data.latest_news(limit=15), use_container_width=True)

        st.subheader("Upload status")
        uploads = data.upload_status()
        u1, u2 = st.columns(2)
        u1.json(uploads.get("by_status", {}))
        u2.json(uploads.get("by_youtube_status", {}))
        st.dataframe(uploads.get("recent", []), use_container_width=True)

        st.subheader("Analytics overview")
        analytics = data.analytics_overview()
        st.json(analytics.get("totals", {}))
        st.dataframe(analytics.get("top_videos", []), use_container_width=True)

        st.subheader("Storage health")
        st.json(data.storage_health())

        st.subheader("System requirements check")
        st.dataframe(data.system_check(), use_container_width=True)

        st.subheader("Recent logs")
        st.dataframe(data.recent_logs(limit=30), use_container_width=True)


if __name__ == "__main__":
    main()
