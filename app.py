import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import os

# ──────────────────────────────────────
# CONFIG
# ──────────────────────────────────────
load_dotenv()

# Get API key: try Streamlit secrets first, then .env, then environment
api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    pass

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("🔑 OpenAI API key not found. Please set it in Streamlit Secrets (Settings → Secrets) as: OPENAI_API_KEY = \"sk-your-key-here\"")
    st.stop()

client = OpenAI(api_key=api_key)

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
# OCR FUNCTION (USES GPT-4O VISION)
# ──────────────────────────────────────
def extract_text_from_image(image: Image.Image) -> str:
    """Send image to GPT-4o Vision for text extraction."""
    
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a precise text extraction engine. Your ONLY job is to extract every visible word from the provided image.

Rules:
- Extract ALL text exactly as it appears. Do not summarize. Do not paraphrase.
- Preserve formatting: line breaks, paragraphs, headers, bullet points.
- If text is rotated, skewed, or at an angle — still extract it accurately.
- If there's handwriting, transcribe it to the best of your ability.
- If there's a table, preserve the table structure using pipes (|) and dashes.
- If there's text in logos, watermarks, headers, footers — extract it all.
- Output ONLY the extracted text. No preamble. No "Here's the text I found." No explanations.
- If you absolutely cannot read something, mark it as [illegible].
- If the image contains no text at all, respond with: [NO TEXT DETECTED]"""
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
        ],
        temperature=0.0,
        max_tokens=4096
    )
    
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF. Uses native extraction first, then OCR for image-based pages."""
    full_text = []
    
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Try native text extraction first
            text = page.extract_text()
            
            if text and text.strip():
                full_text.append(f"--- Page {page_num} ---\n{text}")
            else:
                # Page is likely an image. Convert to image and OCR.
                img = page.to_image(resolution=300)
                pil_image = img.original
                
                with st.spinner(f"🔍 OCR on page {page_num} (scanned image)..."):
                    ocr_text = extract_text_from_image(pil_image)
                    full_text.append(f"--- Page {page_num} (OCR) ---\n{ocr_text}")
    
    return "\n\n".join(full_text)

# ──────────────────────────────────────
# PROCESS UPLOADED FILE
# ──────────────────────────────────────
if uploaded_file:
    file_type = uploaded_file.type
    
    if "pdf" in file_type:
        # PDF Processing
        with st.spinner("📄 Processing PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        # Image Processing
        with st.spinner("👁️ Reading image with AI vision..."):
            image = Image.open(uploaded_file)
            
            # Display the uploaded image
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Uploaded Document", use_container_width=True)
            
            with col2:
                extracted_text = extract_text_from_image(image)
    
    # ── DISPLAY RESULTS ──
    st.divider()
    st.subheader("📋 Extracted Text")
    
    if extracted_text == "[NO TEXT DETECTED]":
        st.warning("No text was detected in this image. It may be a photo without any words, or the text is too blurry to read.")
    else:
        # Stats
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        st.caption(f"Extracted {word_count} words • {char_count} characters")
        
        # Display extracted text
        st.text_area(
            "Extracted Text",
            value=extracted_text,
            height=400,
            key="extracted_text"
        )
        
        # Actions
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.download_button(
                label="📥 Download as .txt",
                data=extracted_text,
                file_name="textsight_extracted.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_b:
            st.caption("📋 To copy: select text above → Cmd+A → Cmd+C")
        with col_c:
            if st.button("🔄 Clear & Upload New", use_container_width=True):
                st.rerun()
        
        # ── INTEGRATION LINKS ──
        st.divider()
        st.markdown("### 🔗 Feed This Text Into Your AI Toolkit")
        
        int_col1, int_col2 = st.columns(2)
        with int_col1:
            st.markdown("""
            **⚖️ Analyze with Adverse Insight**
            
            [Open Adverse Insight](https://adverse-insight.streamlit.app) → Paste the extracted text to analyze the contract for hidden risks and get negotiation scripts.
            """)
        with int_col2:
            st.markdown("""
            **🎣 Scan with PhishTrace**
            
            [Open PhishTrace](https://phishtrace.streamlit.app) → If this is a screenshot of an email, paste the text for phishing forensic analysis.
            """)
        
        # ── DISCLAIMER ──
        st.divider()
        st.caption("Disclaimer: TextSight uses AI vision to extract text. While highly accurate, always verify extracted text against the original document for critical use cases.")

else:
    # ── EMPTY STATE ──
    st.info("👆 Upload an image or PDF to begin text extraction.")
    
    st.divider()
    st.markdown("""
    ### Supported Formats
    
    | Format | Method | Best For |
    |---|---|---|
    | PNG, JPG, WEBP | GPT-4o Vision AI | Screenshots, photos, handwriting, scanned documents |
    | PDF (text) | Direct extraction | Normal PDFs, contracts, reports |
    | PDF (scanned) | Vision AI per page | Scanned books, faxes, image-based PDFs |
    
    ### What makes this different from regular OCR?
    
    Regular OCR struggles with:
    - Rotated or skewed text
    - Handwriting
    - Low-contrast text
    - Mixed fonts and sizes
    - Text in complex backgrounds
    
    TextSight uses GPT-4o Vision — it reads like a human, understanding context and formatting, not just pattern-matching characters.
    """)