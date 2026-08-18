import streamlit as st
import pandas as pd
import re
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
import spacy
import tempfile
import io
import time

# ----------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="RxVision",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Custom CSS — branding / theme
# ----------------------------------------------------------------------
st.markdown("""
    <style>
    /* Overall app background */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Main title */
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1a2a4a;
        margin-bottom: 0;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #5a6b87;
        margin-top: 0.2rem;
    }
    .app-credit {
        font-size: 0.9rem;
        color: #8b98ac;
        font-style: italic;
    }

    /* Section headers */
    h2, h3 {
        color: #1a2a4a !important;
        border-left: 5px solid #3498db;
        padding-left: 10px;
    }

    /* Cards for uploaded results */
    .result-card {
        background-color: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #3498db, #2c80c4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55em 1.4em;
        font-weight: 600;
        transition: 0.2s ease-in-out;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #2c80c4, #1f5f96);
        transform: translateY(-1px);
    }

    /* File uploader box */
    section[data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px dashed #b7c3d6;
    }

    /* Dataframe container */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
col_logo, col_title = st.columns([0.06, 0.94])
with col_logo:
    st.markdown("<div style='font-size:3rem;'>💊</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<div class='app-title'>RxVision - A Doctor Prescription Reader</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Turn prescription photos into structured, downloadable data</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='app-credit'>Built by Kumar Harsh</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------------------
# Sidebar — instructions / about
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About RxVision")
    st.write(
        "Doctors' handwriting can be hard to read — RxVision automatically decodes "
        "handwritten or printed prescriptions and pulls out patient details, so you "
        "don't have to struggle to make sense of it yourself."
    )
    st.markdown("**How to use:**")
    st.markdown(
        "1. Upload one or more prescription images\n"
        "2. Wait for OCR + extraction to finish\n"
        "3. Review the results table\n"
        "4. Download everything as an Excel file"
    )
    st.markdown("---")
    st.caption("Supported formats: JPG, JPEG, PNG")
    st.caption("Made with ❤️ using Streamlit · Kumar Harsh")

    st.markdown("---")
    st.markdown("**📞 Contact**")
    st.markdown("📱 +91 92791 57296")
    st.markdown("✉️ kh949118@gmail.com")

# ----------------------------------------------------------------------
# Cached model loaders (avoid reloading on every rerun)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading OCR engine...")
def load_ocr():
    return PaddleOCR(lang='en')

@st.cache_resource(show_spinner="Loading NLP model...")
def load_nlp():
    return spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "lemmatizer"])

ocr = load_ocr()
nlp = load_nlp()

# ----------------------------------------------------------------------
# Field extraction
# ----------------------------------------------------------------------
def extract_fields(text):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    name = re.search(r"(?:Name|Patient)\s*[:\-]?\s*(.*)", text)
    age = re.search(r"Age\s*[:\-]?\s*(\d+)", text)
    gender = re.search(r"Gender\s*[:\-]?\s*(Male|Female|M|F)", text, re.IGNORECASE)
    return {
        "PatientName": name.group(1).strip() if name else entities.get("PERSON", ""),
        "Age": age.group(1).strip() if age else "",
        "Gender": gender.group(1).strip() if gender else "",
        "Hospital": entities.get("ORG", ""),
        "Doctor": entities.get("PERSON", ""),
        "Date": entities.get("DATE", "")
    }

# ----------------------------------------------------------------------
# Upload section
# ----------------------------------------------------------------------
st.header("📤 Upload Prescriptions")
uploaded_files = st.file_uploader(
    "Drag and drop prescription images here, or browse files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

results = []

if uploaded_files:
    st.header("🔍 Processing")
    progress = st.progress(0, text="Starting...")
    status = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status.info(f"Processing **{uploaded_file.name}** ({i+1}/{len(uploaded_files)})")

        suffix = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        with st.container():
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            img_col, text_col = st.columns([0.35, 0.65])

            if suffix in ["jpg", "jpeg", "png"]:
                image = Image.open(tmp_path)
                image = ImageOps.grayscale(image)
                image = ImageOps.autocontrast(image)
                with img_col:
                    st.image(image, caption=uploaded_file.name, use_container_width=True)

                with st.spinner("Extracting text..."):
                    result = ocr.ocr(tmp_path)
                if result and result[0]:
                    text = "\n".join([line[1][0] for line in result[0]])
                else:
                    text = "No text detected"
            else:
                text = "Unsupported file type"

            fields = extract_fields(text)
            results.append({"Filename": uploaded_file.name, **fields})

            with text_col:
                st.markdown(f"**{uploaded_file.name}**")
                st.text_area("Extracted text", text, height=150, key=f"text_{i}", label_visibility="collapsed")

            st.markdown("</div>", unsafe_allow_html=True)

        progress.progress((i + 1) / len(uploaded_files), text=f"{i+1}/{len(uploaded_files)} processed")

    status.success(f"✅ Done! Processed {len(uploaded_files)} file(s).")
    time.sleep(0.3)
    progress.empty()

    # --------------------------------------------------------------
    # Results table
    # --------------------------------------------------------------
    st.header("📊 Structured Results")
    df = pd.DataFrame(results, columns=[
        "Filename", "PatientName", "Age", "Gender", "Hospital", "Doctor", "Date"
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Files processed", len(df))
    m2.metric("Names detected", int((df["PatientName"] != "").sum()))
    m3.metric("Ages detected", int((df["Age"] != "").sum()))

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------
    st.header("📥 Export")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="RxVision Results")
    output.seek(0)

    st.download_button(
        label="⬇️  Download results as Excel",
        data=output,
        file_name="rxvision_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False
    )
else:
    st.info("👆 Upload one or more prescription images above to get started.")