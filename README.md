# COVID-19 Twitter Sentiment & Emotion — Jupyter Notebook

This project analyzes COVID-19 tweets with:
- **Enhanced EDA**
- **Zero-shot emotion tagging** (pretrained DistilRoBERTa)
- **Fine-tuned DistilBERT** for 5-class sentiment
- **Geospatial mapping** of predictions with Folium

It is designed to run locally with **relative paths** only.

---

## Project structure

```
project/
├─ notebooks/
│  └─ COVID_Twitter_Sentiment_BERT.ipynb
├─ data/                          # place the CSVs here
│  ├─ Corona_NLP_train.csv
│  └─ Corona_NLP_test.csv
├─ artifacts/                     # models, html maps, csvs, etc. (auto-created)
└─ README.md
```

---

## Environment setup

> Python **3.9–3.11** recommended.

> **Torch/GPU**: If you need CUDA builds, install PyTorch following your platform’s instructions, then run `pip install -r requirements.txt` to add the rest.

---

## Data

The two CSVs in are in the folder `./data/`:

- `Corona_NLP_train.csv`
- `Corona_NLP_test.csv`

> Data source: Kaggle — *COVID-19 NLP Text Classification* by **datatattle**.


All file I/O in the notebook uses **relative** paths via:
```python
PATHS["data"]       # -> ./data
PATHS["artifacts"]  # -> ./artifacts
```

---

## 6) What to run (in order)

1. **Load & clean**  
   Creates `clean_text`.

2. **EDA**
   Saves plots/CSV under `./artifacts/` (e.g., `plot_class_counts_by_split.png`, `per_class_stats.csv`).

3. **Emotion tagging (DistilRoBERTa)**  
   - Use the robust cell that **forces** local download if needed (writes to `.hf_cache`).
   - Output map: `artifacts/map_emotions_distilroberta_world.html`.

4. **(Optional) Train** or **load your fine-tuned checkpoint**  
   - To **load** results without retraining, run the single cell:
     ```
     Load a fine-tuned checkpoint (checkpoint-5146)
     ```
     It resolves `./artifacts/results/checkpoint-5146` (or finds it recursively), loads model+tokenizer, and exposes `model`, `tokenizer`, `id2label`.
   - Otherwise set RUN_TRAINING = True to run again the whole training process
   
5. **Evaluation**  
   - Uses the loaded fine-tuned model.
   - Produces the classification report and the confusion matrix image:
     - `artifacts/confusion_matrix_ft.png`
     - `artifacts/test_predictions.csv`

6. **World maps (Folium)**  
   - Ensure the **CITY/COUNTRY patch** cell (expanded `CITY2COUNTRY`, `COUNTRY_CENTROIDS`, `US_STATE_ABBR/NAMES`) is executed **before** mapping.
   - **Predicted vs. Ground-truth** (comparison): saves `artifacts/map_compare_world.html`.
   - **Sentiment-only** or **Emotion-only** maps are also available, saving under `./artifacts/`.

---

## Checkpoints & caching

- Fine-tuned checkpoint expected at:  
  `./artifacts/results/checkpoint-5146/`  
  (contains `config.json`, `pytorch_model.bin`/safetensors, tokenizer files).

- Hugging Face cache is local to the project: `./.hf_cache/` (portable, shareable if needed).

---

## Device usage

The notebook selects device automatically:
- Apple Silicon: **MPS**
- NVIDIA: **CUDA**
- Otherwise: **CPU**

Environment variable already set:  
`PYTORCH_ENABLE_MPS_FALLBACK=1`

---

## Troubleshooting

- **Model won’t load (Hugging Face)**  
  - Ensure internet access for first download.
  - No local folders should be named exactly like a repo (e.g., `j-hartmann/emotion-english-distilroberta-base`) in the CWD — the notebook’s “force download” cell already handles common conflicts.
  - Clear or reuse local cache at `./.hf_cache/`.

- **Maps show few countries**  
  Run the **expanded CITY/COUNTRY patch** cell before mapping. Add missing cities as needed.

- **Noisy or missing locations**  
  The heuristic country inference ignores junk strings and relies on city/state aliases. If you have lat/lon, use the point-based heatmap variant.

---

## Requirements

minimal requirements.txt are be:

```
pandas
numpy
matplotlib
scikit-learn
transformers
huggingface_hub
torch
folium
pyyaml
```

> For GPU/CUDA builds, install the appropriate `torch` wheel before the rest.

---
