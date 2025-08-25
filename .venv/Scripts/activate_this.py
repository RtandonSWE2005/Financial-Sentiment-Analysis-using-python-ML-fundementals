#!/usr/bin/env python3
"""
Financial Text Sentiment Analysis System
========================================
A comprehensive sentiment analysis pipeline for financial news headlines.
Implements data collection, preprocessing, and multiple ML models.

Author: Portfolio Project
Date: 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class FinancialSentimentAnalyzer:
    """
    A comprehensive sentiment analysis system for financial news.
    """

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = TfidfVectorizer()
        self.models = {}
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # Financial-specific stop words to remove
        self.financial_stopwords = {
            'stock', 'stocks', 'market', 'markets', 'trading', 'trade',
            'price', 'prices', 'share', 'shares', 'company', 'companies'
        }
        self.stop_words.update(self.financial_stopwords)

    def collect_yahoo_finance_news(self, symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
                                   max_articles=50):
        """
        Collect news headlines from Yahoo Finance for given stock symbols.
        """
        print("📊 Collecting financial news data...")
        news_data = []

        for symbol in symbols:
            try:
                # Get company info
                ticker = yf.Ticker(symbol)
                news = ticker.news

                print(f"Collecting news for {symbol}...")

                for article in news[:max_articles // len(symbols)]:
                    headline = article.get('title', '')
                    if headline:
                        news_data.append({
                            'symbol': symbol,
                            'headline': headline,
                            'publisher': article.get('publisher', 'Unknown')
                        })

            except Exception as e:
                print(f"Error collecting news for {symbol}: {e}")
                continue

        print(f"✅ Collected {len(news_data)} news headlines")
        return pd.DataFrame(news_data)

    def create_sample_financial_data(self):
        """
        Create sample financial news data with labels for demonstration.
        In a real scenario, you'd have labeled training data.
        """
        sample_data = [
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
            ("Market participants adopt wait-and-see approach", "neutral")
        ]

        return pd.DataFrame(sample_data, columns=['headline', 'sentiment'])

    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing for financial news.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords and apply stemming
        processed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(processed_tokens)

    def prepare_data(self, use_sample_data=True):
        """
        Prepare data for model training.
        """
        print("🔄 Preparing data...")

        if use_sample_data:
            # Use sample data for demonstration
            self.data = self.create_sample_financial_data()
            print("Using sample financial news data for training")
        else:
            # Collect real data from Yahoo Finance
            news_df = self.collect_yahoo_finance_news()

            # Use TextBlob for initial sentiment labeling (weak supervision)
            sentiments = []
            for headline in news_df['headline']:
                blob = TextBlob(headline)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

                sentiments.append(sentiment)

            news_df['sentiment'] = sentiments
            self.data = news_df[['headline', 'sentiment']]

        # Preprocess headlines
        print("📝 Preprocessing text data...")
        self.data['processed_headline'] = self.data['headline'].apply(self.preprocess_text)

        # Remove empty processed headlines
        self.data = self.data[self.data['processed_headline'].str.len() > 0]

        print(f"✅ Data prepared: {len(self.data)} samples")
        print(f"Sentiment distribution:")
        print(self.data['sentiment'].value_counts())

        return self.data

    def create_features(self):
        """
        Create TF-IDF features from preprocessed text.
        """
        print("🔧 Creating TF-IDF features...")

        # Configure TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms appearing in less than 2 documents
            max_df=0.95,  # Ignore terms appearing in more than 95% of documents
            sublinear_tf=True  # Apply sublinear scaling
        )

        # Fit and transform the text data
        X = self.vectorizer.fit_transform(self.data['processed_headline'])
        y = self.data['sentiment']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✅ Features created: {X.shape[1]} features")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")

    def train_models(self):
        """
        Train multiple machine learning models.
        """
        print("🤖 Training machine learning models...")

        # Define models with hyperparameters
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Support Vector Machine': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            }
        }

        # Train each model with grid search
        for name, config in models_config.items():
            print(f"Training {name}...")

            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            grid_search.fit(self.X_train, self.y_train)

            # Store best model
            self.models[name] = grid_search.best_estimator_

            print(f"✅ {name} trained - Best params: {grid_search.best_params_}")

    def evaluate_models(self):
        """
        Evaluate all trained models.
        """
        print("📊 Evaluating models...")

        results = {}

        for name, model in self.models.items():
            print(f"\n--- {name} ---")

            # Predictions
            y_pred = model.predict(self.X_test)

            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)

            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }

            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))

        return results

    def plot_results(self, results):
        """
        Visualize model performance.
        """
        print("📈 Creating visualizations...")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]

        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')

        # Cross-validation comparison
        axes[0, 1].bar(model_names, cv_means, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Cross-Validation Score Comparison')
        axes[0, 1].set_ylabel('CV Score')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(cv_means):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')

        # Sentiment distribution
        sentiment_counts = self.data['sentiment'].value_counts()
        axes[1, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Sentiment Distribution in Dataset')

        # Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_predictions = results[best_model_name]['predictions']

        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1],
                    xticklabels=np.unique(self.y_test),
                    yticklabels=np.unique(self.y_test))
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

        return best_model_name

    def predict_sentiment(self, texts):
        """
        Predict sentiment for new texts using the best model.
        """
        if not self.models:
            raise ValueError("No models trained yet. Please run the full pipeline first.")

        # Get best model
        results = self.evaluate_models()
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = self.models[best_model_name]

        print(f"Using {best_model_name} for predictions...")

        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize
        X_new = self.vectorizer.transform(processed_texts)

        # Predict
        predictions = best_model.predict(X_new)
        probabilities = None

        # Get probabilities if available
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_new)

        return predictions, probabilities

    def run_full_pipeline(self, use_sample_data=True):
        """
        Run the complete sentiment analysis pipeline.
        """
        print("🚀 Starting Financial Sentiment Analysis Pipeline\n")

        # Step 1: Prepare data
        self.prepare_data(use_sample_data=use_sample_data)

        # Step 2: Create features
        self.create_features()

        # Step 3: Train models
        self.train_models()

        # Step 4: Evaluate models
        results = self.evaluate_models()

        # Step 5: Visualize results
        best_model = self.plot_results(results)

        print(f"\n🏆 Best performing model: {best_model}")
        print("✅ Pipeline completed successfully!")

        return results


def main():
    """
    Main function to demonstrate the Financial Sentiment Analyzer.
    """
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()

    # Run full pipeline
    results = analyzer.run_full_pipeline(use_sample_data=True)

    # Test predictions on new headlines
    print("\n🔮 Testing predictions on new headlines:")
    test_headlines = [
        "Apple stock soars to record highs on strong iPhone sales",
        "Market crash wipes billions from tech companies",
        "Federal Reserve maintains current interest rates",
        "Tesla announces major expansion plans",
        "Banking sector faces regulatory challenges"
    ]

    predictions, probabilities = analyzer.predict_sentiment(test_headlines)

    for headline, prediction in zip(test_headlines, predictions):
        print(f"'{headline}' → {prediction}")

    print("\n📚 Pipeline Features Demonstrated:")
    print("✅ Data collection (Yahoo Finance integration)")
    print("✅ Text preprocessing (tokenization, stemming, stopwords)")
    print("✅ TF-IDF feature extraction")
    print("✅ Multiple ML models (Logistic Regression, SVM, Naive Bayes)")
    print("✅ Hyperparameter tuning with Grid Search")
    print("✅ Cross-validation")
    print("✅ Model evaluation and comparison")
    print("✅ Visualization of results")
    print("✅ Production-ready prediction interface")


if __name__ == "__main__":
    main()