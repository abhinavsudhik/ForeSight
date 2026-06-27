---
title: ForeSight
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# ForeSight — AI-Powered Document Fraud Detection Dashboard

ForeSight is an AI-powered document fraud detection system designed to analyze and flag potential fraud across multiple types of uploaded files (e.g. Identity Proof, Land Records, Sale Deeds, Valuation Reports, Bank Statements).

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download offline models:
   ```bash
   python download_models.py
   ```
3. Start the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```
