# Sentiment Analysis Project

This project was built from your reference notebook and upgraded into a reusable sentiment analysis model.

## What it does
- Classifies text as **positive**, **negative**, or **neutral**
- Includes:
  - tokenization
  - stop-word removal
  - lemmatization when NLTK resources are available
  - TF-IDF feature extraction
  - Logistic Regression and Linear SVM
  - accuracy, F1, classification report, confusion matrix
  - model saving for simple deployment

## Files
- `sentiment_analysis_model.py` — train and test the model
- `Sentiment_Analysis_Pro.ipynb` — notebook version
- `sample_data.csv` — small demo dataset
- `requirements.txt` — dependencies

## How to use
1. Put your dataset in the same folder.
2. Set `SENTIMENT_DATA_PATH` or edit `data_path` in the notebook/script.
3. Make sure your data has:
   - one text column
   - one label column

Supported label formats:
- `positive / negative / neutral`
- `1 / 0`
- ratings like `1–5`

## Run
```bash
python sentiment_analysis_model.py
```

## Deployment idea
Save the model and vectorizer with `joblib`, then load them in:
- Streamlit
- Flask
- FastAPI

