import streamlit as st
import base64
import json
from io import BytesIO
from PIL import Image
import pdfplumber
import urllib.request
import urllib.error

# ──────────────────────────────────────
# HARD-CODED API KEY FETCH
# ──────────────────────────────────────
# Read directly from Streamlit secrets with zero abstraction
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("""
    🔑 OpenAI API key not found.
    
    Please add it in Streamlit Cloud:
    1. Go to your app dashboard
    2. Click ⋮ → Settings → Secrets
    3. Add: OPENAI_API_KEY = "sk-your-key-here"
    4. Save and reboot the app
    """)
    st.stop()

# ──────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────
st.set_page_config(
    page_title="TextSight | Universal Text Extractor",
    page_icon="👁️",
    layout="wide"
)

# ──────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────
with st.sidebar:
    st.title("👁️ TextSight")
    st.caption("Universal Document Text Extractor")
    st.divider()
    st.markdown("""
    **What it does:**
    - Extracts text from any image (PNG, JPG, WEBP)
    - Handles PDFs (text-based and scanned)
    - Reads handwriting, screenshots, rotated text
    
    **Use cases:**
    - Screenshot → editable text
    - Photo of a document → copyable text
    - Scanned contract → feed into Adverse Insight
    - Screenshot of phishing email → feed into PhishTrace
    """)
    st.divider()
    st.caption("Built by a Cybersecurity Master's Graduate")
    st.markdown("[GitHub Repo](https://github.com/chima-ukachukwu-sec/textsight)")
    st.divider()
    st.markdown("### Part of the AI Toolkit")
    st.markdown("⚖️ [Adverse Insight](https://adverse-insight.streamlit.app)")
    st.markdown("🎣 [PhishTrace](https://phishtrace.streamlit.app)")
    st.markdown("👁️ TextSight")

# ──────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────
st.title("TextSight")
st.subheader("Extract Text From Any Document")

st.markdown("""
Upload an image or PDF. The AI will extract every word — even from handwriting, screenshots, 
low-quality scans, and rotated text.
""")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
    help="Supports images and PDFs. Max 200MB per file."
)

# ──────────────────────────────────────
# OCR FUNCTION (RAW HTTP TO OPENAI)
# ──────────────────────────────────────
def extract_text_from_image(image: Image.Image) -> str:
    """Send image to GPT-4o Vision using raw HTTP request."""
    
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Build the request payload
    payload = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise text extraction engine. Extract ALL visible text from the image exactly as it appears. Preserve formatting. Output ONLY the extracted text. No explanations. If no text is found, respond with: [NO TEXT DETECTED]"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract ALL text from this image. Preserve formatting exactly."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    }
    
    # Make raw HTTP request
    url = "https://api.openai.com/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise Exception(f"OpenAI API error {e.code}: {error_body}")

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF. Uses native extraction first, then OCR for image-based pages."""
    full_text = []
    
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            
            if text and text.strip():
                full_text.append(f"--- Page {page_num} ---\n{text}")
            else:
                img = page.to_image(resolution=300)
                pil_image = img.original
                
                with st.spinner(f"OCR on page {page_num} (scanned image)..."):
                    ocr_text = extract_text_from_image(pil_image)
                    full_text.append(f"--- Page {page_num} (OCR) ---\n{ocr_text}")
    
    return "\n\n".join(full_text)

# ──────────────────────────────────────
# PROCESS UPLOADED FILE
# ──────────────────────────────────────
if uploaded_file:
    file_type = uploaded_file.type
    
    if "pdf" in file_type:
        with st.spinner("Processing PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        with st.spinner("Reading image with AI vision..."):
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Uploaded Document", use_container_width=True)
            
            with col2:
                extracted_text = extract_text_from_image(image)
    
    # ── DISPLAY RESULTS ──
    st.divider()
    st.subheader("Extracted Text")
    
    if extracted_text == "[NO TEXT DETECTED]":
        st.warning("No text was detected in this image.")
    else:
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        st.caption(f"Extracted {word_count} words • {char_count} characters")
        
        st.text_area(
            "Extracted Text",
            value=extracted_text,
            height=400,
            key="extracted_text"
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.download_button(
                label="Download as .txt",
                data=extracted_text,
                file_name="textsight_extracted.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_b:
            st.caption("To copy: select text above → Cmd+A → Cmd+C")
        with col_c:
            if st.button("Clear & Upload New", use_container_width=True):
                st.rerun()
        
        st.divider()
        st.markdown("### Feed This Text Into Your AI Toolkit")
        
        int_col1, int_col2 = st.columns(2)
        with int_col1:
            st.markdown("""
            **Analyze with Adverse Insight**
            
            [Open Adverse Insight](https://adverse-insight.streamlit.app) → Analyze contracts for hidden risks.
            """)
        with int_col2:
            st.markdown("""
            **Scan with PhishTrace**
            
            [Open PhishTrace](https://phishtrace.streamlit.app) → Forensic analysis of suspicious emails.
            """)
        
        st.divider()
        st.caption("Disclaimer: TextSight uses AI vision to extract text. Always verify against the original document.")

else:
    st.info("Upload an image or PDF to begin text extraction.")
    
    st.divider()
    st.markdown("""
    ### Supported Formats
    
    | Format | Method | Best For |
    |---|---|---|
    | PNG, JPG, WEBP | GPT-4o Vision AI | Screenshots, photos, handwriting |
    | PDF (text) | Direct extraction | Normal PDFs, contracts, reports |
    | PDF (scanned) | Vision AI per page | Scanned books, image-based PDFs |
    """)
