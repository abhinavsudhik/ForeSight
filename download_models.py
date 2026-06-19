#!/usr/bin/env python3
"""
ForeSight — Model Download Script

Downloads and caches all three offline ML models used by the pipeline:
  1. cross-encoder/nli-MiniLM2-L6-H768  (document classification)
  2. deepset/roberta-base-squad2       (field extraction via QA)
  3. Surya OCR (RecognitionPredictor)     (OCR engine)

Run once before first use:
    python download_models.py

All models are cached to ~/.cache/huggingface/ (HuggingFace default).
"""

import sys


def main():
    print("=" * 60)
    print("ForeSight — Downloading Offline ML Models")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. CrossEncoder NLI model (document classification)
    # ------------------------------------------------------------------
    print("\n[1/3] Downloading cross-encoder/nli-MiniLM2-L6-H768 …")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
        print("  ✅ CrossEncoder NLI model downloaded and cached.")
        del model
    except Exception as exc:
        print(f"  ❌ Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. RoBERTa QA model (field extraction)
    # ------------------------------------------------------------------
    print("\n[2/3] Downloading deepset/roberta-base-squad2 …")
    try:
        from transformers import pipeline as hf_pipeline
        qa = hf_pipeline("question-answering", model="deepset/roberta-base-squad2")
        print("  ✅ RoBERTa QA model downloaded and cached.")
        del qa
    except Exception as exc:
        print(f"  ❌ Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Surya OCR models (recognition predictor)
    # ------------------------------------------------------------------
    print("\n[3/3] Downloading Surya OCR models …")
    try:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        print("  Initializing Surya predictor (downloads models on first run) …")
        manager = SuryaInferenceManager()
        predictor = RecognitionPredictor(manager)
        print("  ✅ Surya OCR models downloaded and cached.")
        del predictor, manager
    except Exception as exc:
        print(f"  ❌ Failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("Models are cached in ~/.cache/huggingface/")
    print("You can now run: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
