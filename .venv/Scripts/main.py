#!/usr/bin/env python3
"""
Financial Text Sentiment Analysis System (Extended)
===================================================
End-to-end pipeline for financial news sentiment analysis with:
  • Classical ML (TF–IDF + LR/SVM/NB) with tuning & CV
  • Optional weak labeling of real Yahoo Finance headlines via TextBlob
  • FinBERT (Transformers) for domain-specific sentiment inference
  • Sentiment ↔ stock return correlation analysis & visualization

Author: Portfolio Project
Date: 2025
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
import math
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Core ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Data & labeling
import yfinance as yf
from textblob import TextBlob

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Optional: Transformers (FinBERT)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    """Download minimal NLTK assets if missing."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def _unix_to_date_str(ts: Optional[int]) -> Optional[pd.Timestamp]:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    try:
        # providerPublishTime is seconds since epoch (UTC)
        return pd.to_datetime(int(ts), unit='s').tz_localize('UTC').normalize()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    test_size: float = 0.2
    random_state: int = 42


class FinancialSentimentAnalyzer:
    """A comprehensive sentiment analysis system for financial news."""

    def __init__(self, config: Optional[TrainConfig] = None, verbose: bool = True):
        _ensure_nltk()

        self.verbose = verbose
        self.config = config or TrainConfig()

        # Data containers
        self.data: Optional[pd.DataFrame] = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        # Vectorizer & models
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=True,
        )
        self.models: Dict[str, object] = {}
        self._best_model_name: Optional[str] = None

        # Text preproc
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Finance-specific stopwords to de-emphasize boilerplate
        self.stop_words.update({
            'stock', 'stocks', 'market', 'markets', 'trading', 'trade',
            'price', 'prices', 'share', 'shares', 'company', 'companies'
        })

        # FinBERT
        self.device = torch.device("cuda" if _TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.finbert_tokenizer = None
        self.finbert_model = None
        if _TRANSFORMERS_AVAILABLE:
            try:
                if self.verbose:
                    print("📥 Loading FinBERT (yiyanghkust/finbert-tone)…")
                self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    "yiyanghkust/finbert-tone"
                ).to(self.device)
                self.finbert_model.eval()
                if self.verbose:
                    print("✅ FinBERT loaded.")
            except Exception as e:
                if self.verbose:
                    print(f"⚠ FinBERT failed to load: {e}")
                self.finbert_tokenizer = None
                self.finbert_model = None
        else:
            if self.verbose:
                print("ℹ Transformers not available. FinBERT features disabled.")

    # ----------------------- Data Collection & Labeling ---------------------

    def collect_yahoo_finance_news(self, symbols: List[str] | None = None, max_articles: int = 60) -> pd.DataFrame:
        """Collect recent Yahoo Finance news via yfinance for given symbols.
        Returns columns: [symbol, headline, publisher, published_at (UTC date)]
        """
        symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        rows = []
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                news = t.news or []
                take = max(1, max_articles // max(1, len(symbols)))
                for article in news[:take]:
                    title = (article or {}).get('title')
                    if not title:
                        continue
                    ts = (article or {}).get('providerPublishTime')
                    dt = _unix_to_date_str(ts)
                    rows.append({
                        'symbol': sym,
                        'headline': title,
                        'publisher': (article or {}).get('publisher', 'Unknown'),
                        'published_at': dt
                    })
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Error fetching news for {sym}: {e}")
        df = pd.DataFrame(rows)
        if self.verbose:
            print(f"✅ Collected {len(df)} headlines from Yahoo Finance.")
        return df

    def create_sample_financial_data(self) -> pd.DataFrame:
        data = [
            ("Apple reports record quarterly earnings beating analyst expectations", "positive"),
            ("Tesla stock surges on strong delivery numbers", "positive"),
            ("Microsoft cloud revenue grows significantly", "positive"),
            ("Amazon Web Services continues market dominance", "positive"),
            ("Google parent Alphabet shows robust advertising growth", "positive"),
            ("Strong consumer demand drives tech stock rally", "positive"),
            ("Market reaches new all-time highs on optimism", "positive"),
            ("Investors celebrate strong economic indicators", "positive"),
            ("Corporate profits exceed forecasts across sectors", "positive"),
            ("Bull market continues with sustained growth", "positive"),
            ("Market volatility concerns investors amid uncertainty", "negative"),
            ("Tech stocks decline on regulatory fears", "negative"),
            ("Economic downturn threatens corporate earnings", "negative"),
            ("Supply chain disruptions impact major companies", "negative"),
            ("Interest rate hikes worry market participants", "negative"),
            ("Inflation pressures weigh on consumer spending", "negative"),
            ("Geopolitical tensions create market instability", "negative"),
            ("Corporate layoffs signal economic slowdown", "negative"),
            ("Banking sector faces regulatory scrutiny", "negative"),
            ("Energy prices surge creating economic headwinds", "negative"),
            ("Market closes mixed with no clear direction", "neutral"),
            ("Trading volume remains steady in quiet session", "neutral"),
            ("Analysts maintain hold ratings on major stocks", "neutral"),
            ("Market awaits Federal Reserve decision", "neutral"),
            ("Earnings season begins with modest expectations", "neutral"),
            ("Investors remain cautious ahead of key data", "neutral"),
            ("Market consolidates recent gains", "neutral"),
            ("Trading ranges narrow as volume declines", "neutral"),
            ("Sector rotation continues without clear trend", "neutral"),
            ("Market participants adopt wait-and-see approach", "neutral"),
        ]
        return pd.DataFrame(data, columns=["headline", "sentiment"])

    # ---------------------------- Preprocessing ----------------------------

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        proc = [self.stemmer.stem(tok) for tok in tokens if tok not in self.stop_words and len(tok) > 2]
        return " ".join(proc)

    # -------------------------- Feature Engineering ------------------------

    def prepare_data(self, use_sample_data: bool = True, weak_label_real_news: bool = False,
                     symbols: Optional[List[str]] = None, max_articles: int = 80) -> pd.DataFrame:
        if self.verbose:
            print("🔄 Preparing data…")
        if use_sample_data:
            df = self.create_sample_financial_data()
            if self.verbose:
                print("• Using built-in labeled sample dataset.")
        else:
            df = self.collect_yahoo_finance_news(symbols=symbols, max_articles=max_articles)
            if df.empty:
                raise RuntimeError("No news collected. Try different symbols or increase max_articles.")
            # Weak supervision via TextBlob polarity
            if weak_label_real_news:
                sentiments = []
                for h in df['headline'].astype(str):
                    pol = TextBlob(h).sentiment.polarity
                    if pol > 0.1:
                        sentiments.append('positive')
                    elif pol < -0.1:
                        sentiments.append('negative')
                    else:
                        sentiments.append('neutral')
                df['sentiment'] = sentiments
                df = df[['headline', 'sentiment']]
            else:
                # If no labels, fall back to sample labels for training but keep headlines for inference
                raise ValueError("Real news collected without labels. Set weak_label_real_news=True or provide labeled data.")
        # Preprocess
        df['processed_headline'] = df['headline'].astype(str).apply(self.preprocess_text)
        df = df[df['processed_headline'].str.len() > 0].reset_index(drop=True)
        if self.verbose:
            print(f"✅ Data prepared: {len(df)} samples")
            print("Sentiment distribution:\n" + df['sentiment'].value_counts().to_string())
        self.data = df
        return df

    def create_features(self) -> None:
        if self.data is None:
            raise ValueError("Call prepare_data() first.")
        if self.verbose:
            print("🔧 Creating TF–IDF features…")
        X = self.vectorizer.fit_transform(self.data['processed_headline'])
        y = self.data['sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )
        if self.verbose:
            print(f"• Features: {X.shape[1]} | Train: {self.X_train.shape[0]} | Test: {self.X_test.shape[0]}")

    # ------------------------------ Training -------------------------------

    def train_models(self) -> None:
        if self.X_train is None:
            raise ValueError("Call create_features() first.")
        if self.verbose:
            print("🤖 Training models (GridSearchCV)…")
        configs = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=1000, random_state=self.config.random_state),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs']}
            },
            'Support Vector Machine': {
                'model': SVC(probability=True, random_state=self.config.random_state),
                'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
            },
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {'alpha': [0.1, 0.5, 1.0, 2.0]}
            },
        }
        self.models.clear()
        self._best_model_name = None
        best_acc = -1.0
        for name, cfg in configs.items():
            if self.verbose:
                print(f"• {name}…")
            gs = GridSearchCV(cfg['model'], cfg['params'], cv=5, scoring='accuracy', n_jobs=-1)
            gs.fit(self.X_train, self.y_train)
            self.models[name] = gs.best_estimator_
            # quick holdout acc
            acc = accuracy_score(self.y_test, self.models[name].predict(self.X_test))
            if self.verbose:
                print(f"  ↳ best params: {gs.best_params_} | holdout acc: {acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                self._best_model_name = name
        if self.verbose and self._best_model_name:
            print(f"🏆 Best classical model: {self._best_model_name} ({best_acc:.3f})")

    # ------------------------------ Evaluation ----------------------------

    def evaluate_models(self) -> Dict[str, dict]:
        if not self.models:
            raise ValueError("No trained models. Call train_models().")
        if self.verbose:
            print("📊 Evaluating models…")
        results: Dict[str, dict] = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            results[name] = {
                'accuracy': acc,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'predictions': y_pred,
                'report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            if self.verbose:
                print(f"\n--- {name} ---")
                print(f"Test Accuracy: {acc:.4f}")
                print(f"CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                print(classification_report(self.y_test, y_pred))
        return results

    def plot_results(self, results: Dict[str, dict]) -> str:
        if self.data is None:
            raise ValueError("No data available.")
        # Accuracy comparison
        names = list(results.keys())
        accs = [results[k]['accuracy'] for k in names]
        cv_means = [results[k]['cv_mean'] for k in names]

        plt.figure(figsize=(8, 5))
        plt.bar(names, accs)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.bar(names, cv_means)
        plt.title('Cross-Validation Score Comparison')
        plt.ylabel('CV Score')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        # Sentiment distribution
        plt.figure(figsize=(6, 6))
        counts = self.data['sentiment'].value_counts()
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution in Dataset')
        plt.tight_layout()
        plt.show()

        # Confusion matrix for best model
        best = self._best_model_name or max(results.keys(), key=lambda x: results[x]['accuracy'])
        y_pred = results[best]['predictions']
        cm = confusion_matrix(self.y_test, y_pred, labels=sorted(self.data['sentiment'].unique()))
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest')
        plt.title(f'Confusion Matrix – {best}')
        plt.colorbar()
        tick_marks = np.arange(len(sorted(self.data['sentiment'].unique())))
        plt.xticks(tick_marks, sorted(self.data['sentiment'].unique()), rotation=45)
        plt.yticks(tick_marks, sorted(self.data['sentiment'].unique()))
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
        return best

    # ------------------------------ Inference ------------------------------

    def _get_best_model(self):
        if not self.models:
            raise ValueError("No models trained yet.")
        name = self._best_model_name or max(self.models.keys(), key=lambda k: accuracy_score(self.y_test, self.models[k].predict(self.X_test)))
        return name, self.models[name]

    def predict_sentiment(self, texts: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        name, model = self._get_best_model()
        proc = [self.preprocess_text(t) for t in texts]
        X_new = self.vectorizer.transform(proc)
        preds = model.predict(X_new)
        probs = model.predict_proba(X_new) if hasattr(model, 'predict_proba') else None
        if self.verbose:
            print(f"Using {name} for predictions…")
        return preds, probs

    # --------------------------- FinBERT Inference -------------------------

    def predict_sentiment_finbert(self, texts: List[str]) -> Tuple[List[str], np.ndarray]:
        if self.finbert_model is None or self.finbert_tokenizer is None:
            raise RuntimeError("FinBERT not available. Install transformers/torch or check model load.")
        enc = self.finbert_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.finbert_model(**enc)
            probs = torch.nn.functional.softmax(out.logits, dim=-1)
        # Use model's id2label mapping to avoid hardcoding label order
        id2label = self.finbert_model.config.id2label
        # Some configs may use uppercase labels; normalize to lowercase
        labels = [id2label[int(i)].lower() for i in torch.argmax(probs, dim=1).cpu().numpy()]
        return labels, probs.cpu().numpy()

    # --------------------- Sentiment vs Stock Correlation ------------------

    def sentiment_vs_stock(self, symbol: str = "AAPL", days: int = 60, use_finbert: bool = True,
                           agg: str = 'mode') -> Tuple[pd.DataFrame, float]:
        """Correlate daily sentiment with daily returns for a ticker.
        Returns (merged_df, Pearson correlation).
        """
        if self.verbose:
            print(f"📊 Sentiment vs stock returns for {symbol} (last ~{days} days)…")
        news_df = self.collect_yahoo_finance_news(symbols=[symbol], max_articles=days * 4)
        if news_df.empty:
            raise RuntimeError("No news available for correlation analysis.")
        # Ensure dates exist
        news_df = news_df.dropna(subset=['published_at']).copy()
        if use_finbert:
            labels, _ = self.predict_sentiment_finbert(news_df['headline'].tolist())
            news_df['sentiment'] = labels
        else:
            # Fallback weak labeling
            sentiments = []
            for h in news_df['headline'].astype(str):
                pol = TextBlob(h).sentiment.polarity
                sentiments.append('positive' if pol > 0.1 else ('negative' if pol < -0.1 else 'neutral'))
            news_df['sentiment'] = sentiments
        # Daily aggregate sentiment
        news_df['date'] = news_df['published_at'].dt.tz_convert('UTC').dt.date
        if agg == 'mode':
            daily = news_df.groupby('date')['sentiment'].agg(lambda x: x.mode().iat[0])
        elif agg == 'mean':
            m = {'positive': 1, 'neutral': 0, 'negative': -1}
            daily = news_df.groupby('date')['sentiment'].agg(lambda x: np.sign(np.mean([m.get(v, 0) for v in x])))
            daily = daily.map({1: 'positive', 0: 'neutral', -1: 'negative'})
        else:
            raise ValueError("agg must be 'mode' or 'mean'")
        daily = daily.to_frame('sentiment')
        # Price data
        hist = yf.download(symbol, period=f"{max(5, days+5)}d", interval='1d', progress=False)
        if hist.empty:
            raise RuntimeError("No price data returned.")
        hist['return'] = hist['Close'].pct_change()
        hist.index = pd.to_datetime(hist.index).tz_localize('UTC').date
        merged = daily.join(hist, how='inner')
        # Encode and correlate (same-day; you may also try lead/lag)
        enc = {'positive': 1, 'neutral': 0, 'negative': -1}
        merged['sentiment_score'] = merged['sentiment'].map(enc)
        corr = float(merged['sentiment_score'].corr(merged['return']))
        if self.verbose:
            print(f"📈 Pearson corr(sentiment_score, return): {corr:.3f}")
        # Plot overlay
        plt.figure(figsize=(10, 5))
        plt.plot(merged.index, merged['return'], marker='o', label='Daily Return')
        plt.bar(merged.index, merged['sentiment_score'], alpha=0.4, label='Sentiment Score')
        plt.title(f"{symbol} – Sentiment vs Daily Return")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return merged, corr

    # ----------------------------- Orchestration ---------------------------

    def run_full_pipeline(self, use_sample_data: bool = True, weak_label_real_news: bool = False,
                           symbols: Optional[List[str]] = None, max_articles: int = 80) -> Dict[str, dict]:
        print("🚀 Starting Financial Sentiment Analysis Pipeline\n")
        self.prepare_data(use_sample_data=use_sample_data,
                          weak_label_real_news=weak_label_real_news,
                          symbols=symbols,
                          max_articles=max_articles)
        self.create_features()
        self.train_models()
        results = self.evaluate_models()
        best = self.plot_results(results)
        print(f"\n🏁 Completed. Best model: {best}")
        return results


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def main():
    analyzer = FinancialSentimentAnalyzer()

    # 1) Classical ML pipeline on sample data
    results = analyzer.run_full_pipeline(use_sample_data=True)

    # 2) FinBERT predictions (if available)
    test_headlines = [
        "Apple stock soars to record highs on strong iPhone sales",
        "Market crash wipes billions from tech companies",
        "Federal Reserve maintains current interest rates",
        "Tesla announces major expansion plans",
        "US economy shows resilience amid inflation concerns",
        "Oil prices drop as global demand weakens",
        "Banking sector faces regulatory challenges",
    ]
    if analyzer.finbert_model is not None:
        labels, probs = analyzer.predict_sentiment_finbert(test_headlines)
        print("\n🔮 FinBERT Predictions:")
        for t, lab in zip(test_headlines, labels):
            print(f"'{t}' → {lab}")
    else:
        print("\nℹ FinBERT not available. Install transformers & torch to enable.")

    # 3) Sentiment ↔ stock correlation (uses FinBERT if present)
    try:
        merged, corr = analyzer.sentiment_vs_stock(symbol="AAPL", days=60, use_finbert=(analyzer.finbert_model is not None))
        print("\nMerged sample (head):")
        print(merged[['sentiment', 'return', 'sentiment_score']].head())
        print(f"\nCorrelation: {corr:.3f}")
    except Exception as e:
        print(f"\n⚠ Correlation analysis skipped: {e}")

    # 4) Classical predictor interface demo
    preds, probs = analyzer.predict_sentiment(test_headlines)
    print("\n🤖 Classical Model Predictions:")
    for t, p in zip(test_headlines, preds):
        print(f"'{t}' → {p}")


if __name__ == "__main__":
    main()
