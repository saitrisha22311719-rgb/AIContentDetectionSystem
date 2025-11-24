# app_clean_fixed.py
# Streamlit app: AI-generated text detection + paraphrase (multilingual)
# Clean version without Tools/Debugging section

import streamlit as st
from langdetect import detect, DetectorFactory
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
import os
import torch.nn.functional as F

DetectorFactory.seed = 0

st.set_page_config(
    page_title="AI Generated Text Detector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Generated Text Detection & Paraphrase System")
st.markdown(
    "Predict whether a text is *AI-generated* or *Human-written* (multilingual). "
    "Paraphrasing is offered for AI-generated text."
)

st.sidebar.title("Settings")
local_classifier_path = st.sidebar.text_input("Local classifier path", value="")
classifier_hf_id = st.sidebar.text_input("HF classifier model id", value="bert-base-multilingual-cased")
paraphrase_model_id = st.sidebar.text_input("Paraphrase model id", value="csebuetnlp/mT5_multilingual_XLSum")
max_paraphrase_length = st.sidebar.slider("Paraphrase max length", 32, 512, 256, 16)
min_confidence_for_paraphrase = st.sidebar.slider(
    "Min confidence to trigger paraphrase", 0.50, 0.95, 0.50, 0.05
)
st.sidebar.markdown("⚠ Models may be downloaded on first run if not available locally.")

@st.cache_resource(show_spinner=False)
def load_classifier(local_path: str, fallback_hf: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_from = None
    try:
        if local_path and os.path.exists(local_path):
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_path)
            loaded_from = f"local:{local_path}"
        else:
            tokenizer = AutoTokenizer.from_pretrained(fallback_hf)
            model = AutoModelForSequenceClassification.from_pretrained(fallback_hf)
            loaded_from = f"hub:{fallback_hf}"
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        raise
    model.to(device)
    model.eval()
    return tokenizer, model, device, loaded_from

@st.cache_resource(show_spinner=False)
def load_paraphraser(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.to(device)
    except Exception as e:
        st.warning(f"Paraphraser load failed: {e}")
        return None, None, None
    return tokenizer, model, device

with st.spinner("Loading models..."):
    tokenizer, clf_model, device, loaded_from = load_classifier(local_classifier_path, classifier_hf_id)
    para_tokenizer, para_model, para_device = load_paraphraser(paraphrase_model_id)

st.caption(f"Classifier loaded from: *{loaded_from}* · Device: *{device}*")

def softmax_probs(logits: np.ndarray) -> np.ndarray:
    logits = torch.tensor(logits)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs

def compute_prediction(text: str, min_conf=0.5):
    if not text.strip():
        return None
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        logits = outputs.logits.squeeze().cpu().numpy()

    probs = softmax_probs(logits)
    id2label = (
        clf_model.config.id2label
        if hasattr(clf_model.config, "id2label")
        else {0: "Human", 1: "AI Generated"}
    )
    pred_idx = int(np.argmax(probs))
    label_raw = id2label.get(pred_idx, "AI Generated" if pred_idx == 1 else "Human")
    label = "Human" if "human" in label_raw.lower() else "AI Generated"
    conf = float(probs[pred_idx])

    authenticity = (
        "Very likely AI" if label == "AI Generated" and conf > 0.8 else
        "Mixed AI" if label == "AI Generated" else
        "Very likely Human" if label == "Human" and conf > 0.8 else "Mixed Human"
    )

    all_conf = {}
    for idx, p in enumerate(probs):
        mapped = id2label.get(idx, f"class_{idx}")
        if "human" in mapped.lower():
            all_conf["Human"] = float(p)
        elif "ai" in mapped.lower() or "generated" in mapped.lower():
            all_conf["AI Generated"] = float(p)
        else:
            all_conf[mapped] = float(p)
    all_conf.setdefault("Human", 0.0)
    all_conf.setdefault("AI Generated", 0.0)

    return {
        "prediction": label,
        "confidence": conf,
        "authenticity": authenticity,
        "all_model_confidences": all_conf,
    }

def paraphrase_text(text: str, lang_hint: str = None, max_length: int = 256):
    if para_model is None or para_tokenizer is None:
        return "(paraphraser not available)"
    words = text.split()
    target_len = len(words)
    min_len = max(32, int(target_len * 0.9))
    max_len = int(target_len * 1.1)

    inputs = para_tokenizer(
        text, return_tensors="pt", truncation=True, padding="longest", max_length=512
    ).to(para_model.device)

    with torch.no_grad():
        outputs = para_model.generate(
            **inputs,
            num_beams=8,
            do_sample=True,
            top_p=0.95,
            temperature=0.9,
            min_length=min_len,
            max_length=max_len,
            length_penalty=1.0,
        )
    rephrased = para_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if lang_hint:
        try:
            out_lang = detect(rephrased)
            if out_lang != lang_hint:
                rephrased += f"  (Note: paraphrase language {out_lang} differs from input {lang_hint})"
        except Exception:
            pass
    return rephrased

st.subheader("Enter text to analyze")
text_input = st.text_area("Paste text here", height=220)

uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8")

analyze_btn = st.button("Analyze & Paraphrase if AI")

if analyze_btn:
    if not text_input.strip():
        st.warning("Please provide text.")
    else:
        try:
            lang = detect(text_input)
        except Exception:
            lang = "unknown"

        result = compute_prediction(text_input, min_conf=min_confidence_for_paraphrase)

        st.markdown("### 🔍 Detection Results")
        st.write(f"*Detected language:* {lang}")
        st.write(f"*Prediction:* *{result['prediction']}*")
        st.write(f"*Confidence:* {result['confidence']:.4f}")
        st.write(f"*Authenticity:* {result['authenticity']}")
        st.json(result["all_model_confidences"])

        if result["prediction"] == "AI Generated" and result["confidence"] >= 0.5:
            para_text = paraphrase_text(text_input, lang_hint=lang, max_length=max_paraphrase_length)
            st.markdown("### ✍ Paraphrase")
            st.text_area("Paraphrased text", value=para_text, height=220)
            st.download_button("Download paraphrase (.txt)", data=para_text.encode("utf-8"), file_name="paraphrase.txt")
        else:
            st.info("No paraphrase performed (Human or low confidence).")
