
import streamlit as st
import requests
import base64
import pandas as pd
import plotly.express as px
import sys
import os
import tempfile

from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO

# Make report generator accessible
PROJECT_ROOT = "/content/VisionGuard-AI"
sys.path.insert(0, PROJECT_ROOT)

from backend.report_generator import generate_inspection_report

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🔍",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"
THRESHOLD = 39.40333557128906

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 VisionGuard AI")
st.caption("INDUSTRIAL DEFECT DETECTION & QUALITY INTELLIGENCE")

st.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Inspection Studio", "Quality Analytics"]
)

st.sidebar.divider()
st.sidebar.caption("Powered by PatchCore + PyTorch")

# --------------------------------------------------
# INSPECTION STUDIO
# --------------------------------------------------

if page == "Inspection Studio":

    st.header("Product Inspection")

    uploaded_file = st.file_uploader(
        "Upload an industrial product image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            original_image,
            caption="Original Product",
            use_container_width=True
        )

        if st.button("🔍 Analyse Product", type="primary"):

            with st.spinner("Analysing product..."):

                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        },
                        timeout=180
                    )

                    response.raise_for_status()
                    result = response.json()

                    st.session_state["latest_result"] = result
                    st.session_state["latest_original"] = uploaded_file.getvalue()

                except Exception as error:
                    st.error(f"Inspection failed: {error}")

        # Display the latest result
        if "latest_result" in st.session_state:

            result = st.session_state["latest_result"]

            st.divider()
            st.header("Inspection Results")

            col1, col2 = st.columns(2)

            col1.metric(
                "Anomaly Score",
                f"{result['anomaly_score']:.4f}"
            )

            with col2:
                if result["predicted_defect"]:
                    st.error("⚠️ DEFECTIVE PRODUCT")
                else:
                    st.success("✅ NORMAL PRODUCT")

            st.caption(
                "Classification uses a provisional development "
                "threshold. Independent validation is required."
            )

            heatmap_bytes = base64.b64decode(result["heatmap"])
            overlay_bytes = base64.b64decode(result["overlay"])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Anomaly Heatmap")
                st.image(heatmap_bytes, use_container_width=True)

            with col2:
                st.subheader("Defect Localization")
                st.image(overlay_bytes, use_container_width=True)

            # ----------------------------------
            # PDF REPORT GENERATION
            # ----------------------------------

            st.divider()
            st.subheader("Inspection Report")

            if st.button("Generate PDF Inspection Report"):

                with st.spinner("Generating PDF report..."):

                    with tempfile.TemporaryDirectory() as temp_dir:

                        original_path = os.path.join(temp_dir, "original.png")
                        heatmap_path = os.path.join(temp_dir, "heatmap.png")
                        overlay_path = os.path.join(temp_dir, "overlay.png")
                        pdf_path = os.path.join(temp_dir, "report.pdf")

                        Image.open(
                            BytesIO(st.session_state["latest_original"])
                        ).convert("RGB").save(original_path)

                        Image.open(
                            BytesIO(heatmap_bytes)
                        ).save(heatmap_path)

                        Image.open(
                            BytesIO(overlay_bytes)
                        ).save(overlay_path)

                        report_data = {
                            "filename": result["filename"],
                            "anomaly_score": result["anomaly_score"],
                            "result": result["result"],
                            "threshold": THRESHOLD,
                            "inspected_at": datetime.now().astimezone().isoformat(
                                timespec="seconds"
                            ),
                            "image_width": result["image_width"],
                            "image_height": result["image_height"]
                        }

                        generate_inspection_report(
                            result=report_data,
                            original_path=original_path,
                            heatmap_path=heatmap_path,
                            overlay_path=overlay_path,
                            output_path=pdf_path
                        )

                        with open(pdf_path, "rb") as pdf_file:
                            st.session_state["latest_pdf"] = pdf_file.read()

                st.success("PDF report generated successfully!")

            if "latest_pdf" in st.session_state:

                st.download_button(
                    label="📥 Download Inspection PDF",
                    data=st.session_state["latest_pdf"],
                    file_name="VisionGuard_Inspection_Report.pdf",
                    mime="application/pdf"
                )

# --------------------------------------------------
# QUALITY ANALYTICS
# --------------------------------------------------

elif page == "Quality Analytics":

    st.header("Quality Intelligence Dashboard")

    try:
        response = requests.get(
            f"{API_URL}/history",
            timeout=15
        )

        response.raise_for_status()
        history = response.json()

        total = history["total_inspections"]
        normal = history["normal_products"]
        defective = history["defective_products"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Inspections", total)
        col2.metric("Normal Products", normal)
        col3.metric("Defective Products", defective)
        col4.metric("Defect Rate", f"{history['defect_percentage']:.2f}%")

        records = history["records"]

        if records:

            df = pd.DataFrame(records)

            st.divider()
            st.subheader("Product Quality Distribution")

            distribution = pd.DataFrame({
                "Category": ["Normal", "Defective"],
                "Count": [normal, defective]
            })

            fig = px.pie(
                distribution,
                names="Category",
                values="Count",
                hole=0.55
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Anomaly Score Distribution")

            score_chart = px.histogram(
                df,
                x="anomaly_score",
                color="prediction",
                nbins=20
            )

            st.plotly_chart(score_chart, use_container_width=True)

            st.subheader("Inspection History")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Download Inspection History CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="visionguard_inspection_history.csv",
                mime="text/csv"
            )

        else:
            st.info("No inspection records available.")

    except Exception as error:
        st.error(f"Analytics error: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "VisionGuard AI | PatchCore | Anomalib | PyTorch | FastAPI | Streamlit"
)
