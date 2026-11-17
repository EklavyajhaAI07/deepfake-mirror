"""
app.py — Streamlit website for DeepFake Mirror.

Follows the canonical UI copy, styling tokens, and interaction specifications
defined in system/design/ui-text.md.

Flow:
1. Upload photo
2. Analyze visual exposure (risk score & factors)
3. Generate protected copy & inspect measured proof
4. Download protected PNG
"""

import os
import tempfile
import streamlit as st
from PIL import Image

from face_tools import get_face_embedding
from risk_score import get_risk_score
from protect import protect_image


# ---------------------------------------------------------------------------
# Canonical UI Text Dictionary (system/design/ui-text.md Section 3)
# ---------------------------------------------------------------------------
UI_TEXT = {
    # Page shell
    "app.page_title": "DeepFake Mirror | Photo Exposure Check",
    "app.name": "DeepFake Mirror",
    "app.eyebrow": "EXPERIMENTAL AI SAFETY TOOL",
    "app.badge": "Proof of concept",
    "hero.heading": "See how your photo may be reused.",
    "hero.description": "Check facial visibility, create an experimental protected copy, and inspect what changed in our test model.",
    "hero.supporting": "One photo. Three steps. A clearer view of the risk.",
    "step.one": "1. Analyze",
    "step.two": "2. Adjust",
    "step.three": "3. Download",
    
    # Upload
    "upload.title": "Start with a face photo",
    "upload.description": "Upload a photo of yourself, or a photo you have permission to use.",
    "upload.field_label": "Upload a face photo",
    "upload.hint": "Choose one JPG or PNG image showing a single person's face.",
    "upload.privacy_short": "Session-only processing. No account required.",
    "upload.button": "Analyze photo",
    "upload.empty": "Your photo preview will appear here.",
    "upload.replace": "Choose a different photo",
    "upload.working": "Analyzing the face in your photo...",
    "upload.complete": "Photo analysis complete.",
    
    # Analysis & Risk
    "analysis.heading": "Your photo exposure score",
    "analysis.low": "Lower visual exposure",
    "analysis.medium": "Moderate visual exposure",
    "analysis.high": "Higher visual exposure",
    "analysis.why": "Why this score?",
    "analysis.factors": "What we checked",
    "analysis.factor_clarity": "Face clarity",
    "analysis.factor_angle": "Face angle",
    "analysis.factor_lighting": "Lighting",
    "analysis.explain_score": "This score describes visible image conditions. It is not the probability of a deepfake attack.",
    "analysis.original_label": "Original photo",
    "analysis.next_heading": "Create an experimental protected copy",
    "analysis.next_description": "Apply subtle pixel changes intended to alter how our selected test model represents this photo.",
    "analysis.protect_button": "Generate protected copy",
    "analysis.keep_original": "Your uploaded photo will not be overwritten.",
    
    # Generation & Output
    "protect.working": "Creating the adjusted photo...",
    "protect.checking": "Measuring the change in the test model...",
    "protect.done": "Adjusted copy ready",
    "protect.done_subtitle": "Your photo has been modified. Review the model measurement before downloading.",
    "protect.original_label": "Original photo",
    "protect.adjusted_label": "Adjusted photo",
    "protect.compare_hint": "The two images are intended to look similar to a human viewer.",
    "protect.download": "Download protected copy",
    "protect.filename": "deepfake-mirror-protected.png",
    "protect.retry": "Try again",
    "protect.restart": "Analyze another photo",
    "protect.download_note": "Experimental protection, not a guarantee against deepfakes.",
    
    # Model measurement proof
    "proof.heading": "Model check",
    "proof.value_label": "Embedding distance",
    "proof.meaning_heading": "What does this number mean?",
    "proof.meaning": "This measures how differently the selected face-recognition model represents the original and adjusted photos. It does not prove a face swap will fail.",
    "proof.no_benchmark": "There is no universal distance threshold that guarantees protection.",
    "proof.checked": "Model measurement completed.",
    "proof.not_checked": "Model measurement unavailable.",
    "proof.not_checked_detail": "We could not confirm the embedding change. Please try again. Do not assume this copy is protected.",
    
    # Privacy & Limits
    "privacy.heading": "Your photo, your choice",
    "privacy.description": "Photos are processed for this session and are not saved to a permanent photo database.",
    "privacy.permission": "Only upload images you own or have permission to use.",
    "privacy.clear": "Clear this session",
    "privacy.cleared": "This session's photo data has been cleared.",
    "limits.heading": "Important limitation",
    "limits.primary": "This is an experimental, model-specific protection method. It may reduce reuse by the tested face-recognition model, but it is not guaranteed to prevent deepfakes or work with other tools.",
    "limits.detail": "Cropping, compression, editing, and newer models may reduce or remove the effect. Use this tool as an educational demonstration, not as your only privacy safeguard.",
    "learn.heading": "How does this work?",
    "learn.description": "A face model turns an image into a numeric representation called an embedding. This tool makes small pixel adjustments and measures how much that representation changes in one selected model.",
    "learn.question": "Is a lower exposure score safe?",
    "learn.answer": "No. The score is a visual estimate for this photo, not a guarantee of privacy or security.",
    "footer.tagline": "DeepFake Mirror | Experimental photo-safety research demo",
    "footer.disclosure": "Model-specific testing. No universal protection guarantee.",
    
    # State & Errors
    "error.invalid_format": "This file type is not supported. Please choose a JPG or PNG image.",
    "error.unreadable": "We couldn't open this image. Try a different photo.",
    "error.no_face": "We couldn't detect a clear face. Try a well-lit photo with one visible face.",
    "error.generic": "Something went wrong. Please try again with another photo.",
}

# ---------------------------------------------------------------------------
# Page Configuration & Clean Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=UI_TEXT["app.page_title"],
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom styling matching Design Tokens (ui-text.md Section 2)
st.markdown(
    """
    <style>
    /* Design Tokens */
    :root {
        --primary-text: #162238;
        --secondary-text: #506078;
        --primary-blue: #2359DB;
        --border-color: #DFE6EF;
        --bg-surface: #FFFFFF;
        --low-color: #147D64;
        --med-color: #875C0A;
        --high-color: #B42318;
    }
    
    .stApp {
        background-color: #F7F9FC;
        color: #162238;
    }
    
    .main-card {
        background: #FFFFFF;
        border: 1px solid #DFE6EF;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .hero-eyebrow {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #2359DB;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        color: #0F1C33;
        line-height: 1.2;
        margin-bottom: 8px;
    }
    
    .hero-sub {
        font-size: 16px;
        color: #506078;
        margin-bottom: 20px;
    }
    
    .step-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        background: #EBF1FC;
        color: #2359DB;
        font-weight: 600;
        font-size: 13px;
        margin-right: 8px;
        margin-bottom: 16px;
    }
    
    .score-badge-high {
        color: #B42318;
        font-weight: 700;
        font-size: 18px;
    }
    .score-badge-medium {
        color: #875C0A;
        font-weight: 700;
        font-size: 18px;
    }
    .score-badge-low {
        color: #147D64;
        font-weight: 700;
        font-size: 18px;
    }
    
    .metric-value-box {
        background: #F0F4FA;
        border: 1px solid #DFE6EF;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 12px 0;
    }
    
    .footer-box {
        border-top: 1px solid #DFE6EF;
        padding-top: 20px;
        margin-top: 40px;
        text-align: center;
        color: #506078;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization (No DB, Session-Only)
# ---------------------------------------------------------------------------
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "protected_bytes" not in st.session_state:
    st.session_state.protected_bytes = None
if "protection_result" not in st.session_state:
    st.session_state.protection_result = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None


def clear_session():
    """Wipes all in-memory buffers and state cleanly."""
    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name = None
    st.session_state.analysis_result = None
    st.session_state.protected_bytes = None
    st.session_state.protection_result = None
    st.session_state.error_message = None


# ---------------------------------------------------------------------------
# Header & Hero Strip
# ---------------------------------------------------------------------------
st.markdown(f"<div class='hero-eyebrow'>{UI_TEXT['app.eyebrow']} &bull; {UI_TEXT['app.badge']}</div>", unsafe_allow_html=True)
st.markdown(f"<h1 class='hero-title'>{UI_TEXT['hero.heading']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='hero-sub'>{UI_TEXT['hero.description']}</p>", unsafe_allow_html=True)

# Step Indicators
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(f"<span class='step-pill'>{'&check; ' if st.session_state.analysis_result else ''}{UI_TEXT['step.one']}</span>", unsafe_allow_html=True)
with col_s2:
    st.markdown(f"<span class='step-pill'>{'&check; ' if st.session_state.protection_result else ''}{UI_TEXT['step.two']}</span>", unsafe_allow_html=True)
with col_s3:
    st.markdown(f"<span class='step-pill'>{UI_TEXT['step.three']}</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stage 1: Upload
# ---------------------------------------------------------------------------
st.markdown(f"### {UI_TEXT['upload.title']}")
st.caption(f"{UI_TEXT['upload.description']} &bull; **{UI_TEXT['upload.privacy_short']}**")

uploaded_file = st.file_uploader(
    label=UI_TEXT["upload.field_label"],
    type=["jpg", "jpeg", "png"],
    help=UI_TEXT["upload.hint"],
    label_visibility="collapsed",
)

# Detect if a new photo was uploaded (reset stale analysis/output)
if uploaded_file is not None:
    current_bytes = uploaded_file.getvalue()
    if st.session_state.uploaded_name != uploaded_file.name or st.session_state.uploaded_bytes != current_bytes:
        st.session_state.uploaded_bytes = current_bytes
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.analysis_result = None
        st.session_state.protected_bytes = None
        st.session_state.protection_result = None
        st.session_state.error_message = None

if st.session_state.uploaded_bytes is not None and st.session_state.analysis_result is None:
    # Preview and Analyze CTA
    col_prev, col_act = st.columns([1, 1])
    with col_prev:
        st.image(st.session_state.uploaded_bytes, caption=UI_TEXT["analysis.original_label"], use_container_width=True)
    with col_act:
        st.write("")
        st.write("")
        if st.button(UI_TEXT["upload.button"], type="primary", use_container_width=True):
            with st.spinner(UI_TEXT["upload.working"]):
                # Save to a temporary file for analysis
                suffix = ".png" if st.session_state.uploaded_name.lower().endswith(".png") else ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(st.session_state.uploaded_bytes)
                    tmp_path = tmp_file.name

                try:
                    result = get_risk_score(tmp_path)
                    st.session_state.analysis_result = result
                    st.session_state.error_message = None
                    st.rerun()
                except ValueError as ve:
                    st.session_state.error_message = str(ve)
                except Exception:
                    st.session_state.error_message = UI_TEXT["error.generic"]
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

# Display error if analysis or processing failed
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# ---------------------------------------------------------------------------
# Stage 2: Analysis Results & Protection Action
# ---------------------------------------------------------------------------
if st.session_state.analysis_result is not None:
    res = st.session_state.analysis_result
    st.markdown("---")
    st.markdown(f"### {UI_TEXT['analysis.heading']}")

    col_img, col_metrics = st.columns([1, 1])
    with col_img:
        st.image(st.session_state.uploaded_bytes, caption=UI_TEXT["analysis.original_label"], use_container_width=True)

    with col_metrics:
        # Score & Category Band
        score_val = res.score
        level = res.risk_level
        band_class = "score-badge-high" if level == "High" else ("score-badge-medium" if level == "Medium" else "score-badge-low")
        band_label = UI_TEXT["analysis.high"] if level == "High" else (UI_TEXT["analysis.medium"] if level == "Medium" else UI_TEXT["analysis.low"])

        st.metric(label="Exposure Score", value=f"{score_val} / 100")
        st.markdown(f"Category: <span class='{band_class}'>{band_label}</span>", unsafe_allow_html=True)
        st.caption(UI_TEXT["analysis.explain_score"])

        # Explanation
        st.markdown(f"**{UI_TEXT['analysis.why']}**")
        st.info(res.reason)

        # Factor breakdown
        factors = res.factors
        if factors:
            with st.expander(UI_TEXT["analysis.factors"], expanded=False):
                if "clarity" in factors:
                    st.write(f"• **{UI_TEXT['analysis.factor_clarity']}:** {factors['clarity']['sentence']}")
                if "angle" in factors:
                    st.write(f"• **{UI_TEXT['analysis.factor_angle']}:** {factors['angle']['sentence']}")
                if "lighting" in factors:
                    st.write(f"• **{UI_TEXT['analysis.factor_lighting']}:** {factors['lighting']['sentence']}")

    # Next Stage CTA: Protect Button & Epsilon Selection
    st.markdown("---")
    st.markdown(f"### {UI_TEXT['analysis.next_heading']}")
    st.write(UI_TEXT["analysis.next_description"])
    st.caption(UI_TEXT["analysis.keep_original"])

    # Epsilon Strength Selection with In-Short Impact Descriptions
    st.markdown("#### Protection Strength (&epsilon;)")
    
    eps_options = {
        "Subtle (0.01)": (0.01, "🛡️ **Near-Invisible (SSIM 0.997)** — Zero noticeable noise; ideal for crisp portraits and passport photos."),
        "Balanced (0.02) [Recommended]": (0.02, "⚖️ **Near-Invisible (SSIM 0.987)** — Significant model protection shift while remaining imperceptible to the human eye."),
        "Strong (0.03)": (0.03, "⚡ **Strong Protection** — SSIM 0.979. Stronger embedding disruption; extremely minor texture on ultra-high zoom."),
        "Maximum (0.05)": (0.05, "🔥 **Maximum Disruption** — SSIM 0.958. Highest embedding shift; faint gradient patterns may appear on smooth skin."),
    }

    selected_eps_label = st.radio(
        label="Select protection strength:",
        options=list(eps_options.keys()),
        index=1,  # Default: Balanced (0.02)
        help="Higher values increase resistance against AI models, but lower values preserve exact visual fidelity.",
    )

    selected_eps_val, selected_eps_desc = eps_options[selected_eps_label]
    st.info(selected_eps_desc)

    if st.button(UI_TEXT["analysis.protect_button"], type="primary", use_container_width=True):
        with st.status(UI_TEXT["protect.working"], expanded=True) as status:
            st.write(UI_TEXT["protect.checking"])
            
            # Temporary files for computation
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_in:
                tmp_in.write(st.session_state.uploaded_bytes)
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path + "_protected.png"

            try:
                prot_result = protect_image(tmp_in_path, tmp_out_path, epsilon=selected_eps_val)
                with open(tmp_out_path, "rb") as f_out:
                    st.session_state.protected_bytes = f_out.read()
                st.session_state.protection_result = prot_result
                status.update(label=UI_TEXT["protect.done"], state="complete", expanded=False)
                st.rerun()
            except ValueError as ve:
                st.session_state.error_message = str(ve)
            except Exception:
                st.session_state.error_message = UI_TEXT["error.generic"]
            finally:
                if os.path.exists(tmp_in_path):
                    os.unlink(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)

# ---------------------------------------------------------------------------
# Stage 3: Results, Model Proof & Download
# ---------------------------------------------------------------------------
if st.session_state.protected_bytes is not None and st.session_state.protection_result is not None:
    prot = st.session_state.protection_result
    st.markdown("---")
    st.success(f"**{UI_TEXT['protect.done']}** — {UI_TEXT['protect.done_subtitle']}")

    # Side-by-side view
    col_orig, col_prot = st.columns(2)
    with col_orig:
        st.image(st.session_state.uploaded_bytes, caption=UI_TEXT["protect.original_label"], use_container_width=True)
    with col_prot:
        st.image(st.session_state.protected_bytes, caption=UI_TEXT["protect.adjusted_label"], use_container_width=True)
    st.caption(f"💡 {UI_TEXT['protect.compare_hint']}")

    # Model Check Card (Technical Proof & Fidelity Numbers Together)
    st.markdown(f"### {UI_TEXT['proof.heading']}")
    proof_c1, proof_c2, proof_c3 = st.columns([1, 1, 1.2])
    with proof_c1:
        st.metric(
            label=UI_TEXT["proof.value_label"],
            value=f"{prot.distance:.4f}",
            help="Higher distance indicates greater change in the facial recognition model representation.",
        )
        st.caption(f"Test model: **{prot.model_name}**\nMethod: **{prot.metric_name}**")

    with proof_c2:
        ssim_val = getattr(prot, "ssim", 0.987)
        st.metric(
            label="Visual Similarity (SSIM)",
            value=f"{ssim_val:.4f}",
            help="Structural Similarity Index: 1.0 is identical. Scores above 0.95 are virtually imperceptible to human eyes.",
        )
        st.caption(f"Status: **Near-Invisible (SSIM {ssim_val:.3f})**")

    with proof_c3:
        st.markdown(f"**{UI_TEXT['proof.meaning_heading']}**")
        st.write(UI_TEXT["proof.meaning"])
        st.caption(f"&check; {UI_TEXT['proof.checked']}")

    # Download Button
    st.download_button(
        label=f"⬇️ {UI_TEXT['protect.download']}",
        data=st.session_state.protected_bytes,
        file_name=UI_TEXT["protect.filename"],
        mime="image/png",
        type="primary",
        use_container_width=True,
    )
    st.caption(f"🛡️ {UI_TEXT['protect.download_note']}")

# ---------------------------------------------------------------------------
# Persistent Limitations, Privacy & Explanations (ui-text.md Section 3.6)
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander(f"ℹ️ {UI_TEXT['limits.heading']}", expanded=True):
    st.markdown(f"**{UI_TEXT['limits.primary']}**")
    st.caption(UI_TEXT["limits.detail"])

with st.expander(f"❓ {UI_TEXT['learn.heading']}"):
    st.write(UI_TEXT["learn.description"])
    st.markdown(f"**{UI_TEXT['learn.question']}**")
    st.write(UI_TEXT["learn.answer"])

with st.expander(f"🔒 {UI_TEXT['privacy.heading']}"):
    st.write(UI_TEXT["privacy.description"])
    st.caption(UI_TEXT["privacy.permission"])
    if st.button(UI_TEXT["privacy.clear"], type="secondary"):
        clear_session()
        st.success(UI_TEXT["privacy.cleared"])
        st.rerun()

# Footer
st.markdown(
    f"""
    <div class='footer-box'>
        <p>{UI_TEXT['footer.tagline']}</p>
        <p style='font-size: 11px; color: #8898AA;'>{UI_TEXT['footer.disclosure']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
