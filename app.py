"""
ImageDNA

Screen Capture Detection using
Explainable Machine Learning
"""

from pathlib import Path
import tempfile
import time

import streamlit as st
from PIL import Image

from src.predictor import predict_image

# -------------------------------------------------------
# Page Configuration & Custom CSS Injection
# -------------------------------------------------------

st.set_page_config(
    page_title="ImageDNA | Digital Forensics Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* Base typography & theme overrides */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #e2e8f0;
        }
        
        /* Premium Background Gradient */
        .stApp {
            background: radial-gradient(circle at top right, #1a1c2e 0%, #0c0d16 100%);
        }
        
        /* Custom Cards */
        .forensic-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(8px);
        }
        
        /* Metric Styling */
        .metric-title {
            font-size: 0.9rem;
            color: #94a3b8;
            font-weight: 500;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Result Badges */
        .result-badge {
            padding: 1rem 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 0.02em;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .badge-camera {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
        }
        .badge-screen {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fbbf24;
            box-shadow: 0 0 15px rgba(245, 158, 11, 0.1);
        }
        
        /* Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: #ffffff !important;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 14px rgba(3, 105, 161, 0.3);
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(3, 105, 161, 0.5);
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        }
        div.stButton > button:first-child:active {
            transform: translateY(0);
        }
        
        /* Forensic Indicators List */
        .indicator-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.6rem 0.8rem;
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid #f59e0b;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
        }
        .indicator-success {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.8rem 1rem;
            background: rgba(16, 185, 129, 0.05);
            border-left: 3px solid #10b981;
            border-radius: 0 8px 8px 0;
            color: #a7f3d0;
            font-size: 0.95rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>🔬 ImageDNA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Digital Image Forensics Laboratory</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### System Profile")
    st.markdown("""
        <div class='forensic-card' style='padding: 1rem; margin-bottom: 1rem;'>
            <table style='width: 100%; font-size: 0.85rem;'>
                <tr>
                    <td style='color: #94a3b8; padding-bottom: 4px;'>Model Engine</td>
                    <td style='text-align: right; font-weight: 600; padding-bottom: 4px;'>Random Forest</td>
                </tr>
                <tr>
                    <td style='color: #94a3b8; padding-bottom: 4px;'>Cross-Validation</td>
                    <td style='text-align: right; font-weight: 600; padding-bottom: 4px;'>5-Fold Stratified</td>
                </tr>
                <tr>
                    <td style='color: #94a3b8; padding-bottom: 4px;'>Model Accuracy</td>
                    <td style='text-align: right; font-weight: 600; color: #10b981; padding-bottom: 4px;'>84.19%</td>
                </tr>
                <tr>
                    <td style='color: #94a3b8;'>Avg. Latency</td>
                    <td style='text-align: right; font-weight: 600; color: #38bdf8;'>~20 ms (Laptop CPU)</td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Main Header
# -------------------------------------------------------

st.title("🔬 ImageDNA Analysis Suite")
st.markdown("<p style='font-size: 1.1rem; color: #94a3b8; margin-top: -0.5rem;'>Verify whether an image is a genuine camera capture or a photograph of a digital display.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------
# Upload Section & File Details
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Select an image file for forensic screening:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([5, 3])
    
    with col1:
        st.image(image, caption="Loaded Target Image", use_container_width=True)
        
    with col2:
        st.markdown("### Image Properties")
        st.markdown(f"""
            <div class='forensic-card'>
                <table style='width: 100%; border-collapse: collapse; font-size: 0.95rem;'>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 0.8rem 0; color: #94a3b8;'>File Name</td>
                        <td style='padding: 0.8rem 0; text-align: right; font-weight: 600; word-break: break-all;'>{uploaded_file.name}</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 0.8rem 0; color: #94a3b8;'>Dimensions</td>
                        <td style='padding: 0.8rem 0; text-align: right; font-weight: 600;'>{image.size[0]} × {image.size[1]} px</td>
                    </tr>
                    <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <td style='padding: 0.8rem 0; color: #94a3b8;'>Color Profile</td>
                        <td style='padding: 0.8rem 0; text-align: right; font-weight: 600;'>{image.mode}</td>
                    </tr>
                    <tr>
                        <td style='padding: 0.8rem 0; color: #94a3b8;'>Aspect Ratio</td>
                        <td style='padding: 0.8rem 0; text-align: right; font-weight: 600;'>{round(image.size[0]/image.size[1], 2)}:1</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        
        analyze = st.button("Run Forensic Analysis", use_container_width=True)
else:
    analyze = False

# -------------------------------------------------------
# Perform Prediction
# -------------------------------------------------------

if analyze:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    with st.spinner("Extracting forensic descriptors and running verification..."):
        start_time = time.perf_counter()
        result = predict_image(temp_path)
        latency_ms = (time.perf_counter() - start_time) * 1000

    prediction = result["prediction"]
    confidence = result["confidence"]
    evidence = result["evidence"]

    st.markdown("---")
    st.subheader("📊 Forensic Screening Report")

    # Layout for Badge & Confidence
    col_results1, col_results2 = st.columns([5, 5])

    with col_results1:
        st.markdown("<p style='font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;'>Verdict</p>", unsafe_allow_html=True)
        if prediction == "Real Photo":
            st.markdown("""
                <div class='result-badge badge-camera'>
                    <span>📷</span> Original Camera Capture
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-badge badge-screen'>
                    <span>💻</span> Screen Capture / Display Photo
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='font-size: 0.9rem; color: #94a3b8; margin-top: -0.5rem; display: flex; align-items: center; gap: 6px;'>
                <span>⏱️</span> <span>Inference Latency:</span> 
                <span style='color: #38bdf8; font-weight: 600;'>{latency_ms:.1f} ms</span> 
                <span style='color: #64748b; font-size: 0.8rem;'>(Laptop CPU)</span>
            </div>
        """, unsafe_allow_html=True)

    with col_results2:
        # Map numerical confidence to descriptive rating
        if confidence >= 90:
            level = "High Certainty"
            level_color = "#10b981"
        elif confidence >= 75:
            level = "Moderate Certainty"
            level_color = "#3b82f6"
        elif confidence >= 60:
            level = "Reasonable Match"
            level_color = "#f59e0b"
        else:
            level = "Low Borderline Match"
            level_color = "#ef4444"

        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: flex-end;'>
                <div>
                    <div class='metric-title'>Confidence Score</div>
                    <div class='metric-value'>{confidence:.2f}%</div>
                </div>
                <div style='text-align: right; padding-bottom: 0.2rem;'>
                    <span style='color: #94a3b8; font-size: 0.85rem;'>Assessment:</span><br>
                    <span style='color: {level_color}; font-weight: 700; font-size: 1.1rem;'>{level}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.progress(confidence / 100)

    # Detailed Indicators & Explanations Layout
    col_det1, col_det2 = st.columns([5, 5])

    with col_det1:
        st.markdown("### Forensic Evidence Log")
        if evidence:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.9rem;'>The following structural deviations were detected:</p>", unsafe_allow_html=True)
            for item in evidence:
                st.markdown(f"""
                    <div class='indicator-item'>
                        <span style='color: #f59e0b; font-weight: bold;'>⚠️</span> {item}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='indicator-success'>
                    <span>✓</span> All examined parameters are consistent with native camera captures. No display grid distortions detected.
                </div>
            """, unsafe_allow_html=True)

    with col_det2:
        st.markdown("### Expert Summary")
        if prediction == "Real Photo":
            summary_text = "Our statistical and frequency analysis indicates that this image is a **native camera capture**. The pixel layout, frequency decay profile, and noise statistics align cleanly with real-world sensor captures and show no signs of display moiré distortion or digital screenshot interpolation."
        else:
            summary_text = "Our analysis strongly suggests this image represents a **photograph of a digital display** or a direct screenshot. This conclusion is driven by signature artifacts like distorted frequency energy profiles, unnatural edge distributions, and the lack of standard camera sensor noise residuals in flat areas."
        
        st.markdown(f"""
            <div class='forensic-card' style='font-size: 0.95rem; line-height: 1.6; color: #cbd5e1;'>
                {summary_text}
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 0.8rem; line-height: 1.6;'>
            Developed by Divya Bansal | B.Tech Computer Science (Data Science) | Bennett University
        </div>
    """, unsafe_allow_html=True)

    # Delete temporary image
    temp_path.unlink(missing_ok=True)