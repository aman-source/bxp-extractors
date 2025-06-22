import streamlit as st
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from utils.db_utils import get_finetune_files, save_file_record
from utils.file_utils import save_uploaded_file
from utils.azure_finetune import train_model_on_azure

# Load environment variables
load_dotenv()
DEFAULT_MODEL = os.getenv("ACTIVE_MODEL", "mistral")
AZURE_CONTAINER_SAS_URL = os.getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_SAS_URL")

st.set_page_config(page_title="Bank Invoice Extractor", layout="wide")
st.title("Bank Invoice Extractor")

# Session state to retain values
st.session_state.setdefault("extracted_json", None)
st.session_state.setdefault("uploaded_file", None)

uploaded_file = st.file_uploader("Upload scanned invoice (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])

model_options = ["mistral", "azure", "datalab", "anthropic", "gemini"]
DEFAULT_MODEL = st.selectbox("Choose Model", model_options, index=0)
doc_type = st.radio("Document Type", ["Invoice", "Bank Statement"], horizontal=True)

st.markdown(f"**Active model:** `{DEFAULT_MODEL}`")

# --- Extraction ---
if st.button("Extract Information"):
    if uploaded_file is None:
        st.warning("Please upload a file first.")
    else:
        saved_path = save_uploaded_file(uploaded_file)

        with st.spinner(f"Processing with {DEFAULT_MODEL.title()} OCR..."):
            if DEFAULT_MODEL == "mistral":
                from models.mistral_ocr_handler import extract_from_mistral
                result = extract_from_mistral(saved_path, doc_type=doc_type)
            elif DEFAULT_MODEL == "azure":
                from models.azure_handler import extract_from_azure
                result = extract_from_azure(saved_path, doc_type=doc_type)
            elif DEFAULT_MODEL == "datalab":
                from models.datalab_handler import extract_from_datalab
                result = extract_from_datalab(saved_path, doc_type=doc_type)
            elif DEFAULT_MODEL == "anthropic":
                from models.anthropic_handler import extract_from_claude
                result = extract_from_claude(saved_path, doc_type=doc_type)
            elif DEFAULT_MODEL == "gemini":
                from models.gemini_handler import extract_from_gemini
                result = extract_from_gemini(saved_path, doc_type=doc_type)
            else:
                result = {"error": f"Unknown model: {DEFAULT_MODEL}"}

        # Set session state for reuse
        st.session_state["extracted_json"] = result
        st.session_state["uploaded_file"] = uploaded_file

# Show extracted JSON if available
if st.session_state.get("extracted_json"):
    st.subheader("Extracted JSON")
    st.json(st.session_state["extracted_json"])

# --- Validation & Save as "success" ---
st.markdown("---")
st.subheader("Validation Framework")

st.markdown("**Paste the expected output JSON below:**")
expected_json_text = st.text_area("Expected Output JSON", height=300)

if st.button("Run Validation Framework (All Models)"):
    if st.session_state.get("uploaded_file") is None or not expected_json_text.strip():
        st.warning("Please upload a file and paste the expected JSON.")
    else:
        saved_path = save_uploaded_file(st.session_state["uploaded_file"])

        from validation.validation_framework import validate_all_models

        try:
            expected_json = json.loads(expected_json_text)
            with st.spinner("Running all models and evaluating metrics..."):
                df = validate_all_models(saved_path, expected_json)

            st.success("Validation complete")
            st.subheader("Model Comparison Metrics")
            st.dataframe(df)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

# --- Fine-tune flag section ---
if st.button("Mark as Needs Fine-Tune"):
    if st.session_state.get("uploaded_file") is None:
        st.warning("Upload a file before flagging.")
    elif not expected_json_text.strip():
        st.warning("Provide expected JSON before flagging.")
    else:
        try:
            expected_json = json.loads(expected_json_text)
            st.session_state["uploaded_file"].seek(0)
            url = save_file_record(
                file=st.session_state["uploaded_file"],
                filename=st.session_state["uploaded_file"].name,
                model=DEFAULT_MODEL,
                doc_type=doc_type,
                expected_json=expected_json,
                status="needs_finetune"
            )
            if url:
                st.success("Document marked as 'needs fine-tune'.")
                st.markdown(f"[📄 View Uploaded File]({url})")
            else:
                st.warning("Save failed — no URL returned.")
        except json.JSONDecodeError:
            st.error("Expected JSON is invalid.")

# --- Fine-Tune Management Section ---
st.markdown("---")
st.subheader(" Fine-Tune Management")

files = get_finetune_files()
count = len(files)
print(files)  # Debugging output
st.info(f"{count} files marked as 'needs fine-tune'")


if count > 0:
    if st.button("Trigger Fine-Tune Now"):
        training_inputs = []

        for f in files:
            try:
                # Parse the expectedJson safely
                parsed_json = json.loads(f.get("expectedJson", "{}").replace("```json", "").replace("```", "").strip())
                training_inputs.append({
                    "docId": f["docId"],
                    "fileUrl": f["fileUrl"],
                    "expectedJson": parsed_json,
                    "docType": f["docType"],
                    "model": f["model"]
                })
            except json.JSONDecodeError as e:
                st.warning(f"Skipping docId {f.get('docId')} due to invalid JSON.")
        
        if not training_inputs:
            st.error("No valid training files with parsable expected JSON.")
        else:
            with st.spinner("Triggering Azure Document Intelligence training..."):
                from os import getenv
                sas_url = getenv("DOCUMENTINTELLIGENCE_STORAGE_CONTAINER_SAS_URL")
                model_id = train_model_on_azure(
                    container_sas_url=sas_url,
                    description=f"Fine-tuned model with {len(training_inputs)} docs from external API"
                )

            if model_id:
                st.success(f"Fine-tune triggered successfully. Model ID: `{model_id}`")
                st.markdown("✅ You can view this model in Azure AI Studio.")
            else:
                st.error("Failed to trigger Azure training.")
else:
    st.info("No files available for fine-tuning.")
