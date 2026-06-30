"""
ImageDNA

Screen Capture Detection using
Explainable Machine Learning
"""

from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.predictor import predict_image


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(

    page_title="ImageDNA",

    page_icon="",

    layout="wide",

    initial_sidebar_state="expanded"

)


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("ImageDNA")

    st.markdown("---")

    st.header("About")

    st.write(
        """
ImageDNA is a digital image forensics system
that determines whether an image is

• Original Camera Photo

or

• Screen Capture

using handcrafted forensic features and
Explainable Machine Learning.
"""
    )

    st.markdown("---")

    st.header("Model")

    st.write("Random Forest Classifier")

    st.write("20 Forensic Features")

    st.write("5-Fold Cross Validation")

    st.write("Accuracy : **82.75%**")

    st.markdown("---")

    st.header("Forensic Features")

    st.write("• Sharpness")

    st.write("• FFT Analysis")

    st.write("• High Frequency Ratio")

    st.write("• Noise Statistics")

    st.write("• Edge Density")

    st.write("• LBP Texture")

    st.write("• GLCM Texture")

    st.write("• Wavelets")

    st.write("• JPEG Blocking")


# -------------------------------------------------------
# Main Header
# -------------------------------------------------------

st.title("ImageDNA")

st.subheader(
    "Screen Capture Detection using Explainable Machine Learning"
)

st.write(
    """
Upload an image and ImageDNA will analyze its
forensic characteristics to determine whether
it is an original camera photograph or
a screen-captured image.
"""
)

st.markdown("---")


# -------------------------------------------------------
# Upload Section
# -------------------------------------------------------

uploaded_file = st.file_uploader(

    "Upload an Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)


# -------------------------------------------------------
# Preview
# -------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([2, 1])

    with col1:

        st.image(

            image,

            caption="Uploaded Image",

            use_container_width=True

        )

    with col2:

        st.info(

            f"""
Filename

{uploaded_file.name}

Resolution

{image.size[0]} × {image.size[1]}

Mode

{image.mode}
"""

        )

    st.markdown("---")

    analyze = st.button(

        "Analyze Image",

        use_container_width=True

    )

else:

    analyze = False








# -------------------------------------------------------
# Perform Prediction
# -------------------------------------------------------

if analyze:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpeg"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())

        temp_path = Path(temp_file.name)

    with st.spinner("Performing forensic analysis..."):

        result = predict_image(temp_path)

    prediction = result["prediction"]

    confidence = result["confidence"]

    evidence = result["evidence"]

    st.markdown("---")

    st.header("📄 Forensic Analysis Report")

    # -------------------------------------------------------
    # Prediction Card
    # -------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if prediction == "Real Photo":

            st.success("📷 REAL CAMERA PHOTO")

        else:

            st.error("💻 SCREEN CAPTURE")

    with col2:

        if confidence >= 90:

            level = "High"

        elif confidence >= 75:

            level = "Moderate"

        elif confidence >= 60:

            level = "Reasonable"

        else:

            level = "Low"

        st.metric(
            "Classification Confidence",
            f"{confidence:.2f}%"
        )

        st.write(f"Confidence Level : **{level}**")

    # -------------------------------------------------------
    # Confidence Meter
    # -------------------------------------------------------

    st.subheader("Confidence Meter")

    st.progress(confidence / 100)

    # -------------------------------------------------------
    # Forensic Indicators
    # -------------------------------------------------------

    st.subheader("Forensic Indicators")

    if evidence:

        for item in evidence:

            st.write(f"✓ {item}")

    else:

        st.success(
            "No suspicious forensic indicators detected."
        )

    # -------------------------------------------------------
    # Final Decision
    # -------------------------------------------------------

    st.subheader("Conclusion")

    if prediction == "Real Photo":

        st.info(
            """
The uploaded image is more likely to be an
original camera photograph.

The forensic characteristics are closer to
those observed in genuine camera-captured
images than screen recaptured images.
"""
        )

    else:

        st.warning(
            """
The uploaded image is more likely to be
captured from a digital display.

Several forensic characteristics resemble
screen recaptured images.
"""
        )

    # -------------------------------------------------------
    # Footer
    # -------------------------------------------------------

    st.markdown("---")

    st.caption(
        """
ImageDNA v1.0

Developed by Divya Bansal

B.Tech Computer Science (Data Science)

Bennett University
"""
    )

    # Delete temporary image

    temp_path.unlink(missing_ok=True)