📊 Financial Sentiment Analyzer (Python, ML + FinBERT)

This project is a Financial Text Sentiment Analyzer built in Python, combining traditional ML models (TF-IDF + Logistic Regression, SVM, Naive Bayes) with state-of-the-art NLP (FinBERT).
It also performs stock correlation analysis, linking sentiment in financial headlines to real market returns.

🚀 Features

Classical ML Pipeline

TF-IDF vectorization

Logistic Regression / SVM / Naive Bayes with GridSearchCV

Evaluation: accuracy, F1, confusion matrix, ROC curves

FinBERT Integration

Transformer model fine-tuned for financial sentiment

Predicts sentiment on sample and real news headlines

GPU-accelerated if CUDA is available

Stock Price Correlation

Scrapes financial headlines from Yahoo Finance (via yfinance)

Aggregates daily sentiment scores

Visualizes sentiment vs return correlation

Visualization

Confusion matrix heatmaps

ROC curves for each model

Sentiment vs stock return scatter plots
