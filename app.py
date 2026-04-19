import streamlit as st
import pandas as pd
import os
import io

st.set_page_config(page_title="Log Classifier", page_icon="📋", layout="wide")

st.title("📋 Hybrid Log Classification System")
st.markdown("Upload a CSV with `source` and `log_message` columns to classify logs using Regex → BERT → LLM pipeline.")

# ── Sidebar: API Key ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Required for LegacyCRM logs (LLM classification)")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("API key set!")
    else:
        st.warning("Enter your Groq API key for LLM classification.")

    st.divider()
    st.header("📖 Pipeline")
    st.markdown("""
    1. **Regex** — fast rule-based matching  
    2. **BERT** — sentence embeddings + logistic regression  
    3. **LLM (Groq)** — for LegacyCRM logs only  
    """)
    st.divider()
    st.header("📁 CSV Format")
    st.markdown("Your CSV must have these columns:")
    st.code("source, log_message")
    sample = pd.DataFrame({
        "source": ["ModernCRM", "LegacyCRM"],
        "log_message": ["Backup completed successfully.", "Case escalation for ticket ID 7324 failed."]
    })
    st.dataframe(sample, use_container_width=True)

# ── Main Area ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    if "source" not in df.columns or "log_message" not in df.columns:
        st.error("❌ CSV must contain both `source` and `log_message` columns.")
    else:
        st.subheader("📄 Uploaded Data")
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            run_btn = st.button("🚀 Classify Logs", type="primary", use_container_width=True)

        if run_btn:
            # Check if LegacyCRM rows exist and API key is missing
            has_legacy = "LegacyCRM" in df["source"].values
            if has_legacy and not groq_api_key:
                st.error("❌ Your CSV contains LegacyCRM logs which require the Groq API key. Please enter it in the sidebar.")
            else:
                with st.spinner("Classifying logs... this may take a moment for LLM calls."):
                    try:
                        from classify import classify
                        labels = classify(list(zip(df["source"], df["log_message"])))
                        df["target_label"] = labels

                        st.success("✅ Classification complete!")

                        # ── Results ────────────────────────────────────────────
                        st.subheader("📊 Results")

                        # Summary metrics
                        label_counts = df["target_label"].value_counts()
                        cols = st.columns(min(len(label_counts), 4))
                        for i, (label, count) in enumerate(label_counts.items()):
                            with cols[i % 4]:
                                st.metric(label, count)

                        st.divider()

                        # Color-coded table
                        def color_label(val):
                            colors = {
                                "Workflow Error": "background-color: #ffcccc",
                                "Deprecation Warning": "background-color: #fff3cc",
                                "User Action": "background-color: #cce5ff",
                                "System Notification": "background-color: #ccffcc",
                                "Unclassified": "background-color: #e0e0e0",
                            }
                            return colors.get(val, "")

                        styled = df.style.applymap(color_label, subset=["target_label"])
                        st.dataframe(styled, use_container_width=True)

                        # ── Download ───────────────────────────────────────────
                        csv_out = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Results CSV",
                            data=csv_out,
                            file_name="classified_output.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except ImportError as e:
                        st.error(f"❌ Import error: {e}\nMake sure all processor files are in the same directory.")
                    except Exception as e:
                        st.error(f"❌ Classification failed: {e}")

else:
    # Show example when no file uploaded
    st.info("👆 Upload a CSV file to get started.")
    st.subheader("💡 Example Input")
    example_df = pd.DataFrame({
        "source": ["ModernCRM", "BillingSystem", "LegacyCRM", "AnalyticsEngine", "ModernHR"],
        "log_message": [
            "IP 192.168.133.114 blocked due to potential attack",
            "User User12345 logged in.",
            "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
            "Backup completed successfully.",
            "Admin access escalation detected for user 9429"
        ]
    })
    st.dataframe(example_df, use_container_width=True)

st.divider()
st.markdown("<div style='text-align:center;color:#888;'>Hybrid Log Classifier · Regex → BERT → LLM</div>", unsafe_allow_html=True)
