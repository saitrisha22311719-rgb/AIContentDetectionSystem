# 🧠 AI Content Detection System (Streamlit App)

This project detects **AI-generated text** vs **Human-written text** using **Multilingual BERT (mBERT)**, with multilingual **paraphrasing** using **mT5**.

## 🚀 Features
- Detects if text is **AI-generated** or **Human-written**
- Supports multiple languages
- Uses **mBERT** for classification
- Paraphrases AI-generated text using **mT5**
- Streamlit web interface for interactive use

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

## 🧪 Run the Streamlit App
```bash
streamlit run app_clean_fixed.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

## 📦 Project Structure
```
AIContentDetectionSystem_Streamlit/
├── aicontentdetectionsystem.py    # Core training and prediction logic
├── app_clean_fixed.py             # Streamlit web app
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .gitignore
```

## 📜 License
MIT License
