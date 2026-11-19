# SMS Spam Detection

A machine-learning pipeline that classifies SMS messages as **spam** or **ham** (not spam). It compares Multinomial Naive Bayes and Logistic Regression on TF-IDF features and saves the better model for prediction.

## Dataset
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): about 5,500 labelled English SMS messages (roughly 13% spam). The script downloads it automatically on the first run, or you can place `SMSSpamCollection` (UCI) or `spam.csv` (Kaggle) in the `data/` folder.

## Approach
1. **Text cleaning** (`preprocess.py`): lowercase, replace URLs and numbers with tokens, remove punctuation, remove stopwords, Porter stemming.
2. **Features**: TF-IDF with unigrams and bigrams (`sublinear_tf=True`).
3. **Models**: Multinomial Naive Bayes and Logistic Regression (`class_weight="balanced"`).
4. **Evaluation**: stratified 80/20 train/test split. Accuracy, precision, recall, F1 and confusion matrix are reported, because accuracy alone is misleading on imbalanced data.
5. The best model by spam F1-score is saved to `models/spam_model.joblib`.

## Results
Run `python train.py` and copy your own numbers here (also saved in `results/metrics.json`):

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|
| Naive Bayes | _fill_ | _fill_ | _fill_ | _fill_ |
| Logistic Regression | _fill_ | _fill_ | _fill_ | _fill_ |

## Project structure
```
sms-spam-detection/
├── preprocess.py      # text cleaning
├── train.py           # train, compare, evaluate, save model
├── predict.py         # classify new messages
├── requirements.txt
├── data/              # dataset (downloaded automatically)
├── models/            # saved model
└── results/           # metrics.json
```

## How to run
```bash
git clone https://github.com/AnkitaMaurya63/sms-spam-detection.git
cd sms-spam-detection
pip install -r requirements.txt
python train.py
python predict.py "Congratulations! You won a free prize, call now"
```

## Web app and REST API
After training, start the FastAPI server:
```bash
uvicorn app:app --reload
```
- Web UI: http://127.0.0.1:8000 (paste a message, get spam/not-spam, probability and the words that raised the score)
- API docs: http://127.0.0.1:8000/docs
- Endpoint: `POST /predict` with `{"text": "your message"}` returns `{"label", "spam_probability", "spam_signals"}`

## Tech stack
Python, scikit-learn, pandas, NLTK, joblib, FastAPI, HTML/CSS/JavaScript.

## Possible improvements
- Hyperparameter tuning with cross-validation
- Try SVM or a transformer-based classifier
- Serve the model with FastAPI
