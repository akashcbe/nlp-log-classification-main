import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Log Classifier", page_icon="📋", layout="wide")

st.title("📋 Hybrid Log Classification System")
st.markdown("Upload a CSV with `source` and `log_message` columns to classify logs using Regex → BERT → LLM pipeline.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.text_input(
        "Groq API Key",
        type="password",
        key="groq_api_key",
        help="Required for LegacyCRM logs"
    )
    if st.session_state.groq_api_key:
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
        st.success("✅ API key saved!")

    st.divider()
    st.header("📖 Pipeline")
    st.markdown("""
    1. **Regex** — fast rule-based matching  
    2. **BERT** — sentence embeddings + logistic regression  
    3. **LLM (Groq)** — for LegacyCRM logs only  
    """)
    st.divider()
    st.header("📁 CSV Format")
    st.code("source, log_message")

# ── Main ───────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "source" not in df.columns or "log_message" not in df.columns:
        st.error("❌ CSV must contain both `source` and `log_message` columns.")
        st.stop()

    st.subheader("📄 Uploaded Data")
    st.dataframe(df, width='stretch')

    has_legacy = "LegacyCRM" in df["source"].values
    api_key_ready = bool(st.session_state.get("groq_api_key", ""))

    if has_legacy and not api_key_ready:
        st.warning("⚠️ This CSV has LegacyCRM rows. Please enter your Groq API key in the sidebar.")

    run_btn = st.button(
        "🚀 Classify Logs",
        type="primary",
        disabled=(has_legacy and not api_key_ready)
    )

    if run_btn:
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key

        with st.spinner("Classifying... LLM calls may take a moment."):
            try:
                from classify import classify
                labels = classify(list(zip(df["source"], df["log_message"])))
                df["target_label"] = labels

                st.success("✅ Classification complete!")

                label_counts = df["target_label"].value_counts()
                cols = st.columns(min(len(label_counts), 4))
                for i, (label, count) in enumerate(label_counts.items()):
                    with cols[i % 4]:
                        st.metric(label, count)

                st.divider()

                def color_label(val):
                    return {
                        "Workflow Error":      "background-color: #ffcccc",
                        "Deprecation Warning": "background-color: #fff3cc",
                        "User Action":         "background-color: #cce5ff",
                        "System Notification": "background-color: #ccffcc",
                        "Security Alert":      "background-color: #f3ccff",
                        "HTTP Log":            "background-color: #ffe0cc",
                        "Unclassified":        "background-color: #e0e0e0",
                    }.get(val, "")

                st.dataframe(
                    df.style.map(color_label, subset=["target_label"]),
                    width='stretch'
                )

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="classified_output.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Classification failed: {e}")

else:
    st.info("👆 Upload a CSV file to get started.")
    st.dataframe(pd.DataFrame({
        "source": ["ModernCRM", "BillingSystem", "LegacyCRM", "AnalyticsEngine"],
        "log_message": [
            "IP 192.168.133.114 blocked due to potential attack",
            "User User12345 logged in.",
            "Case escalation for ticket ID 7324 failed.",
            "Backup completed successfully.",
        ]
    }), width='stretch')

st.divider()
st.markdown("<div style='text-align:center;color:#888;'>Hybrid Log Classifier · Regex → BERT → LLM</div>",
            unsafe_allow_html=True)
