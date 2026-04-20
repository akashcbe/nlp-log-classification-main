import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Log Classifier", page_icon="📋", layout="wide")

# Store API key in session state so it persists across reruns
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

st.title("📋 Hybrid Log Classification System")
st.markdown("Upload a CSV with `source` and `log_message` columns to classify logs using Regex → BERT → LLM pipeline.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Required for LegacyCRM logs (LLM classification)"
    )
    if key_input:
        st.session_state.groq_api_key = key_input
        os.environ["GROQ_API_KEY"] = key_input
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
    st.markdown("Your CSV must have:")
    st.code("source, log_message")

# ── Main ───────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "source" not in df.columns or "log_message" not in df.columns:
        st.error("❌ CSV must contain both `source` and `log_message` columns.")
        st.stop()

    st.subheader("📄 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    has_legacy = "LegacyCRM" in df["source"].values
    api_key_ready = bool(st.session_state.groq_api_key)

    if has_legacy and not api_key_ready:
        st.warning("⚠️ This CSV has LegacyCRM rows. Please enter your Groq API key in the sidebar.")

    run_btn = st.button(
        "🚀 Classify Logs",
        type="primary",
        disabled=(has_legacy and not api_key_ready)
    )

    if run_btn:
        # Ensure env var is set before classify runs
        os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key

        with st.spinner("Classifying logs... this may take a moment for LLM calls."):
            try:
                from classify import classify
                labels = classify(list(zip(df["source"], df["log_message"])))
                df["target_label"] = labels

                st.success("✅ Classification complete!")

                st.subheader("📊 Results")
                label_counts = df["target_label"].value_counts()
                cols = st.columns(min(len(label_counts), 4))
                for i, (label, count) in enumerate(label_counts.items()):
                    with cols[i % 4]:
                        st.metric(label, count)

                st.divider()

                def color_label(val):
                    colors = {
                        "Workflow Error":      "background-color: #ffcccc",
                        "Deprecation Warning": "background-color: #fff3cc",
                        "User Action":         "background-color: #cce5ff",
                        "System Notification": "background-color: #ccffcc",
                        "Unclassified":        "background-color: #e0e0e0",
                    }
                    return colors.get(val, "")

                styled = df.style.map(color_label, subset=["target_label"])
                st.dataframe(styled, use_container_width=True)

                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="classified_output.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Classification failed: {e}")

else:
    st.info("👆 Upload a CSV file to get started.")
    st.subheader("💡 Example Input")
    st.dataframe(pd.DataFrame({
        "source": ["ModernCRM", "BillingSystem", "LegacyCRM", "AnalyticsEngine"],
        "log_message": [
            "IP 192.168.133.114 blocked due to potential attack",
            "User User12345 logged in.",
            "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
            "Backup completed successfully.",
        ]
    }), use_container_width=True)

st.divider()
st.markdown("<div style='text-align:center;color:#888;'>Hybrid Log Classifier · Regex → BERT → LLM</div>",
            unsafe_allow_html=True)
