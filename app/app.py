"""
PCB-VLM-XAI Streamlit Application
====================================
PURPOSE:
    Main entry point for the web-based PCB inspection interface.
    Provides a multi-page app with upload, detection, XAI, retrieval,
    knowledge base, and inspection report pages.

USAGE:
    streamlit run app/app.py
    streamlit run app/app.py --server.port 8501 --server.maxUploadSize 50
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import yaml
import numpy as np
import cv2
from PIL import Image
from loguru import logger

# ============================================================
# Page configuration (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="PCB-VLM-XAI | Intelligent PCB Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Load configuration
# ============================================================
@st.cache_resource
def load_app_config():
    config_path = PROJECT_ROOT / "configs" / "streamlit.yaml"
    try:
        with open(config_path, encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


config = load_app_config()

# ============================================================
# Custom CSS for dark theme
# ============================================================
st.markdown("""
<style>
    /* Main app dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Severity badges */
    .severity-critical {
        background-color: #ff4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .severity-high {
        background-color: #ff8800;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .severity-medium {
        background-color: #ffcc00;
        color: black;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    .severity-low {
        background-color: #44ff44;
        color: black;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    
    /* Detection box */
    .detection-box {
        background: #16213e;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #e0e0e0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #0f3460, #533483);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    /* Info boxes */
    .info-box {
        background: #1a1a2e;
        border: 1px solid #533483;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* Progress bar color */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0f3460, #533483);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Pipeline initialization (cached across sessions)
# ============================================================
@st.cache_resource(show_spinner="Loading inspection models...")
def load_pipeline():
    """Load the complete inspection pipeline. Cached globally."""
    from pipeline.orchestrator import PCBInspectionPipeline
    pipeline = PCBInspectionPipeline(
        config_dir=str(PROJECT_ROOT / "configs"),
        enable_xai=True,
        enable_retrieval=True,
        enable_llm=True,
    )
    return pipeline


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge_engine():
    from knowledge.knowledge_engine import KnowledgeEngine
    return KnowledgeEngine(str(PROJECT_ROOT / "knowledge" / "defect_knowledge.json"))


# ============================================================
# Session state initialization
# ============================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "pipeline_result": None,
        "uploaded_image": None,
        "uploaded_filename": "",
        "current_page": "Upload & Detect",
        "analysis_complete": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ============================================================
# Sidebar Navigation
# ============================================================
def render_sidebar():
    """Render the sidebar navigation and settings."""
    with st.sidebar:
        st.markdown("## 🔬 PCB-VLM-XAI")
        st.markdown("*Explainable Vision-Language PCB Inspection*")
        st.markdown("---")

        pages = [
            ("📷", "Upload & Detect"),
            ("🔍", "XAI Visualizations"),
            ("🔎", "Similar Defects"),
            ("📚", "Knowledge Insights"),
            ("📋", "Inspection Report"),
        ]

        st.markdown("### Navigation")
        selected_page = None
        for icon, page_name in pages:
            is_active = st.session_state.current_page == page_name
            btn_label = f"{icon} {page_name}"
            
            # Disable non-upload pages if no analysis done
            disabled = not st.session_state.analysis_complete and page_name != "Upload & Detect"
            
            if st.button(
                btn_label,
                use_container_width=True,
                disabled=disabled,
                key=f"nav_{page_name}",
                type="primary" if is_active else "secondary",
            ):
                selected_page = page_name

        if selected_page:
            st.session_state.current_page = selected_page
            st.rerun()

        st.markdown("---")

        # Status indicators
        st.markdown("### System Status")
        result = st.session_state.pipeline_result

        if result and result.success:
            st.success(f"✅ {result.num_defects} defects found")
            st.info(f"⏱️ {result.processing_time:.1f}s total")

            if result.stage_times:
                with st.expander("Timing breakdown"):
                    for stage, t in result.stage_times.items():
                        st.text(f"{stage}: {t*1000:.0f}ms")
        else:
            st.info("No image analyzed yet")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Models:**
        - 🎯 YOLOv8s (Detection)
        - 🌡️ Grad-CAM (XAI)
        - 🔍 SigLIP + FAISS (Retrieval)
        - 🤖 Qwen2.5-1.5B LoRA (Reports)
        
        **Hardware Target:**
        RTX 4050 6GB · i7-13650HX · 24GB RAM
        """)


# ============================================================
# Page: Upload & Detect
# ============================================================
def render_upload_page():
    """Render the upload and detection page."""
    st.markdown("# 📷 Upload PCB & Detect Defects")
    st.markdown("Upload a PCB image to begin automated defect inspection.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a PCB image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a PCB image for defect inspection",
        )

        if uploaded_file:
            # Store in session state
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.uploaded_image = image
            st.session_state.uploaded_filename = uploaded_file.name

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption=f"Uploaded: {uploaded_file.name}",
                use_column_width=True,
            )

            h, w = image.shape[:2]
            st.caption(f"Resolution: {w} × {h} pixels")

    with col2:
        st.markdown("### Detection Settings")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9,
            value=0.35, step=0.05,
            help="Minimum confidence to report a defect",
        )

        enable_xai = st.checkbox("Generate Grad-CAM Heatmaps", value=True)
        enable_retrieval = st.checkbox("Retrieve Similar Cases", value=True)
        enable_llm = st.checkbox("Generate LLM Report", value=True)

        if enable_llm:
            st.warning(
                "⚠️ LLM report generation may take 30-60 seconds on first run "
                "while the model loads into memory."
            )

        st.markdown("---")

        run_button = st.button(
            "🚀 Run Inspection",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.uploaded_image is None,
        )

        if run_button and st.session_state.uploaded_image is not None:
            _run_inspection(
                st.session_state.uploaded_image,
                conf_threshold=conf_threshold,
                enable_xai=enable_xai,
                enable_retrieval=enable_retrieval,
                enable_llm=enable_llm,
            )

    # Show results if available
    if st.session_state.analysis_complete and st.session_state.pipeline_result:
        _render_detection_results()


def _run_inspection(image, conf_threshold, enable_xai, enable_retrieval, enable_llm):
    """Run the inspection pipeline and update session state."""
    with st.spinner("🔬 Running PCB inspection pipeline..."):
        pipeline = load_pipeline()

        # Override confidence threshold
        try:
            pipeline._get_detector().conf_threshold = conf_threshold
        except Exception:
            pass

        # Progress tracking
        progress = st.progress(0, text="Detecting defects...")

        result = pipeline.run(
            image=image,
            run_xai=enable_xai,
            run_retrieval=enable_retrieval,
            run_llm=enable_llm,
        )
        progress.progress(100, text="Complete!")

        st.session_state.pipeline_result = result
        st.session_state.analysis_complete = True

    if result.success:
        st.success(f"✅ Inspection complete! Found {result.num_defects} defect(s)")
    else:
        st.error(f"❌ Inspection failed: {result.error_message}")

    st.rerun()


def _render_detection_results():
    """Render detection results after analysis."""
    result = st.session_state.pipeline_result
    if not result:
        return

    st.markdown("---")
    st.markdown("## Detection Results")

    knowledge_engine = load_knowledge_engine()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Defects", result.num_defects)
    with col2:
        critical = sum(1 for d in result.detections
                      if knowledge_engine.get_severity(d.class_name) == "critical")
        st.metric("Critical", critical, delta=f"{'⚠️' if critical else '✅'}")
    with col3:
        if result.detections:
            avg_conf = np.mean([d.confidence for d in result.detections])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    with col4:
        st.metric("Inference Time", f"{result.stage_times.get('detection', 0)*1000:.0f}ms")

    # Detection image
    if result.detection_image is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                cv2.cvtColor(result.detection_image, cv2.COLOR_BGR2RGB),
                caption="Detection Results",
                use_column_width=True,
            )
        with col2:
            st.markdown("### Detected Defects")
            if not result.detections:
                st.success("No defects detected!")
            else:
                for det in result.detections:
                    severity = knowledge_engine.get_severity(det.class_name)
                    color_map = {
                        "critical": "🔴", "high": "🟠",
                        "medium": "🟡", "low": "🟢"
                    }
                    icon = color_map.get(severity, "⚪")
                    with st.expander(
                        f"{icon} {det.class_name.replace('_', ' ').title()} ({det.confidence:.1%})"
                    ):
                        st.markdown(f"**Severity:** {severity.upper()}")
                        st.markdown(f"**Class ID:** {det.class_id}")
                        st.markdown(f"**Bbox:** {[int(v) for v in det.bbox]}")

                        causes = knowledge_engine.get_causes(det.class_name)
                        if causes:
                            st.markdown("**Top Cause:**")
                            st.markdown(f"*{causes[0]}*")


# ============================================================
# Page: XAI Visualizations
# ============================================================
def render_xai_page():
    """Render the XAI/Grad-CAM visualization page."""
    st.markdown("# 🔍 XAI Visualizations")
    st.markdown(
        "Grad-CAM heatmaps show **which regions** the neural network focused on "
        "when detecting defects. Warmer colors (red/yellow) indicate higher activation."
    )

    result = st.session_state.pipeline_result
    if not result or result.heatmap is None:
        st.warning("No XAI data available. Enable 'Generate Grad-CAM Heatmaps' and run inspection.")
        return

    # Full XAI panel
    if result.xai_panel is not None:
        st.markdown("### Full Analysis Panel")
        st.image(
            cv2.cvtColor(result.xai_panel, cv2.COLOR_BGR2RGB),
            caption="Original | Detections | Heatmap | CAM Overlay",
            use_column_width=True,
        )

    # Individual views
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Grad-CAM Heatmap")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#0e1117")
        im = ax.imshow(result.heatmap, cmap="jet", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Activation Strength")
        ax.set_title("Grad-CAM Activation Map", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("### CAM Overlay")
        if result.cam_overlay is not None:
            st.image(
                cv2.cvtColor(result.cam_overlay, cv2.COLOR_BGR2RGB),
                caption="Heatmap blended with original PCB",
                use_column_width=True,
            )

    # Download button
    if result.cam_overlay is not None:
        overlay_rgb = cv2.cvtColor(result.cam_overlay, cv2.COLOR_BGR2RGB)
        pil_overlay = Image.fromarray(overlay_rgb)
        import io
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Overlay Image",
            data=buf.getvalue(),
            file_name="gradcam_overlay.png",
            mime="image/png",
        )

    # Educational explanation
    with st.expander("📖 How does Grad-CAM work?"):
        st.markdown("""
        **Gradient-weighted Class Activation Mapping (Grad-CAM)**
        
        1. **Forward pass**: Run the image through the neural network
        2. **Target class**: Select which class to explain (e.g., "missing_hole")
        3. **Gradient computation**: Calculate how much each feature map activation 
           contributes to the class score using backpropagation
        4. **Weight features**: Average gradients spatially to get importance weights
        5. **Weighted sum**: Multiply each feature map by its importance weight
        6. **Apply ReLU**: Keep only positive contributions
        7. **Resize & overlay**: Scale heatmap to input image size and blend
        
        **Result**: Red areas = network focuses here | Blue areas = less relevant
        """)


# ============================================================
# Page: Similar Defects
# ============================================================
def render_retrieval_page():
    """Render the similar defects retrieval gallery."""
    st.markdown("# 🔎 Similar Defect Retrieval")
    st.markdown(
        "For each detected defect, the system retrieves the **3 most visually similar** "
        "cases from the historical database using SigLIP embeddings + FAISS search."
    )

    result = st.session_state.pipeline_result
    if not result or not result.retrieved_cases:
        st.warning("No retrieval data. Enable 'Retrieve Similar Cases' and run inspection.")
        return

    for det_idx, similar_cases in result.retrieved_cases.items():
        if det_idx >= len(result.detections):
            continue

        det = result.detections[det_idx]
        st.markdown(f"### Defect: {det.class_name.replace('_', ' ').title()} ({det.confidence:.1%})")

        if not similar_cases:
            st.info("No similar cases found above similarity threshold.")
            continue

        # Show crop + similar cases
        cols = st.columns(min(len(similar_cases) + 1, 4))

        with cols[0]:
            st.markdown("**Query Crop**")
            if det.crop is not None:
                st.image(
                    cv2.cvtColor(det.crop, cv2.COLOR_BGR2RGB),
                    caption="Detected defect",
                    use_column_width=True,
                )

        for i, case in enumerate(similar_cases[:3]):
            with cols[i + 1]:
                similarity = case.get("similarity", 0)
                label = case.get("label", "unknown").replace("_", " ").title()

                # Similarity color
                if similarity >= 0.85:
                    sim_color = "🟢"
                elif similarity >= 0.7:
                    sim_color = "🟡"
                else:
                    sim_color = "🟠"

                st.markdown(f"**Similar #{i+1}**")
                st.markdown(f"{sim_color} Similarity: **{similarity:.2f}**")
                st.markdown(f"Class: *{label}*")

                # Show crop image if path exists
                crop_path = case.get("crop_path", "")
                if crop_path and Path(crop_path).exists():
                    img = cv2.imread(crop_path)
                    if img is not None:
                        st.image(
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            use_column_width=True,
                        )
                else:
                    st.caption("(Image not available)")

        st.markdown("---")

    with st.expander("📖 How does similarity search work?"):
        st.markdown("""
        **SigLIP + FAISS Retrieval Pipeline**
        
        1. **Crop**: Extract defect region from PCB image (+20px padding)
        2. **SigLIP**: Pass crop through SigLIP vision encoder → 768-dim vector
        3. **Normalize**: L2-normalize vector (enables cosine similarity via dot product)
        4. **FAISS Search**: Find nearest neighbors in 768-dim embedding space
        5. **Rank**: Sort results by cosine similarity score (1.0 = identical)
        
        **Why SigLIP?**
        SigLIP is trained on billions of image-text pairs, so it understands
        visual semantics — not just pixel patterns. Two "missing hole" defects
        from different PCBs will be close in embedding space even if they differ
        in exact appearance.
        """)


# ============================================================
# Page: Knowledge Insights
# ============================================================
def render_knowledge_page():
    """Render the knowledge base insights page."""
    st.markdown("# 📚 Knowledge Insights")
    st.markdown("Domain expert knowledge about each detected defect type.")

    knowledge_engine = load_knowledge_engine()
    result = st.session_state.pipeline_result

    if not result or not result.detections:
        # Show all defect knowledge
        st.markdown("*No defects detected. Showing complete knowledge base:*")
        defect_classes = knowledge_engine.get_all_defect_classes()
    else:
        defect_classes = list(set(d.class_name for d in result.detections))

    selected_defect = st.selectbox(
        "Select Defect Type",
        options=defect_classes,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if selected_defect:
        info = knowledge_engine.get_defect_info(selected_defect)
        if not info:
            st.warning(f"No knowledge found for: {selected_defect}")
            return

        # Severity badge
        severity = info.get("severity", "unknown")
        sev_colors = {
            "critical": "#ff4444", "high": "#ff8800",
            "medium": "#ffcc00", "low": "#44ff44"
        }
        sev_color = sev_colors.get(severity, "#888888")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div style='background:{sev_color}; padding:8px; border-radius:8px; "
                f"text-align:center; font-weight:bold; color:{'white' if severity in ['critical','high'] else 'black'}'>"
                f"Severity: {severity.upper()}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("Category", info.get("category", "N/A").title())
        with col3:
            st.metric("IPC Code", info.get("code", "N/A"))

        st.markdown(f"### {info['name']}")
        st.markdown(info.get("description", ""))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔧 Manufacturing Causes")
            causes = info.get("causes", [])
            for cause in causes:
                st.markdown(f"- {cause}")

            st.markdown("#### ⚠️ Potential Risks")
            risks = info.get("potential_risks", [])
            for risk in risks:
                st.markdown(f"- {risk}")

        with col2:
            st.markdown("#### 🔍 Inspection Procedure")
            procs = info.get("inspection_procedure", [])
            for i, proc in enumerate(procs, 1):
                st.markdown(f"{i}. {proc}")

            st.markdown("#### 🛠️ Repair Recommendations")
            recs = info.get("repair_recommendations", [])
            for rec in recs:
                st.markdown(f"- {rec}")

        # Manufacturing process
        mfg = info.get("manufacturing_process", {})
        if mfg:
            with st.expander("🏭 Manufacturing Process Details"):
                st.markdown(f"**Stage:** {mfg.get('stage', 'N/A')}")
                st.markdown(mfg.get("process_description", ""))
                factors = mfg.get("contributing_factors", [])
                if factors:
                    st.markdown("**Contributing Factors:**")
                    for f in factors:
                        st.markdown(f"- {f}")

        st.markdown(f"*IPC Standard: {info.get('ipc_standard', 'N/A')}*")


# ============================================================
# Page: Inspection Report
# ============================================================
def render_report_page():
    """Render the LLM-generated inspection report."""
    st.markdown("# 📋 Inspection Report")
    st.markdown("AI-generated expert inspection report powered by fine-tuned Qwen2.5.")

    result = st.session_state.pipeline_result
    if not result:
        st.warning("No inspection data available. Upload and analyze a PCB first.")
        return

    if not result.report:
        st.warning("No report generated. Enable 'Generate LLM Report' and run inspection.")
        return

    # Report display
    st.markdown("---")
    st.markdown(result.report)
    st.markdown("---")

    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=result.report,
            file_name=f"pcb_inspection_report_{st.session_state.uploaded_filename}.md",
            mime="text/markdown",
        )
    with col2:
        # JSON export
        import json
        summary_json = json.dumps(result.get_summary(), indent=2)
        st.download_button(
            label="⬇️ Download Summary (JSON)",
            data=summary_json,
            file_name="inspection_summary.json",
            mime="application/json",
        )


# ============================================================
# Main router
# ============================================================
def main():
    """Main app entry point."""
    render_sidebar()

    page = st.session_state.current_page

    if page == "Upload & Detect":
        render_upload_page()
    elif page == "XAI Visualizations":
        render_xai_page()
    elif page == "Similar Defects":
        render_retrieval_page()
    elif page == "Knowledge Insights":
        render_knowledge_page()
    elif page == "Inspection Report":
        render_report_page()
    else:
        render_upload_page()


if __name__ == "__main__":
    main()
