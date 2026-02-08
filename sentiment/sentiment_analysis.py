"""
Sentiment Analysis Module
MSc IT Project - Stock Price Forecasting

This module performs sentiment analysis on financial news headlines
using NLTK and TextBlob
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class SentimentAnalyzer:
    """Analyze sentiment of financial news"""
    
    def __init__(self):
        """Initialize Sentiment Analyzer"""
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        
        # Financial sentiment keywords
        self.positive_words = {
            'surge', 'soar', 'gain', 'profit', 'growth', 'bullish', 'upgrade',
            'beat', 'exceed', 'breakthrough', 'innovation', 'strong', 'positive',
            'rally', 'rise', 'jump', 'climb', 'advance', 'outperform'
        }
        
        self.negative_words = {
            'fall', 'drop', 'decline', 'loss', 'bearish', 'downgrade', 'miss',
            'concern', 'worry', 'weak', 'negative', 'plunge', 'crash', 'slump',
            'tumble', 'sink', 'underperform', 'fail', 'cut', 'reduce'
        }
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Join back to string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not isinstance(text, str) or len(text) == 0:
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral'
            }
        
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get polarity and subjectivity
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        if polarity > 0.05:
            label = 'Positive'
        elif polarity < -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_score': polarity,
            'sentiment_label': label
        }
    
    def enhance_sentiment_financial(self, text, base_sentiment):
        """
        Enhance sentiment score using financial keywords
        
        Args:
            text (str): Original text
            base_sentiment (dict): Base sentiment from TextBlob
            
        Returns:
            dict: Enhanced sentiment scores
        """
        if not isinstance(text, str):
            return base_sentiment
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count positive and negative financial words
        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        
        # Calculate financial sentiment boost
        total_words = len(words)
        if total_words > 0:
            financial_sentiment = (pos_count - neg_count) / total_words
        else:
            financial_sentiment = 0
        
        # Combine with TextBlob sentiment (weighted average)
        enhanced_score = (0.6 * base_sentiment['polarity']) + (0.4 * financial_sentiment)
        
        # Update sentiment label based on enhanced score
        if enhanced_score > 0.1:
            label = 'Positive'
        elif enhanced_score < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
        
        return {
            'polarity': base_sentiment['polarity'],
            'subjectivity': base_sentiment['subjectivity'],
            'sentiment_score': enhanced_score,
            'sentiment_label': label,
            'financial_keywords_pos': pos_count,
            'financial_keywords_neg': neg_count
        }
    
    def analyze_news_dataframe(self, news_df):
        """
        Analyze sentiment for all news articles in dataframe
        
        Args:
            news_df (pd.DataFrame): News dataframe with 'headline' column
            
        Returns:
            pd.DataFrame: News dataframe with sentiment scores
        """
        print(f"Analyzing sentiment for {len(news_df)} news articles...")
        
        sentiment_results = []
        
        for idx, row in news_df.iterrows():
            # Combine headline and description for better analysis
            text = str(row.get('headline', ''))
            if 'description' in row and pd.notna(row['description']):
                text += ' ' + str(row['description'])
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Get base sentiment
            base_sentiment = self.analyze_sentiment_textblob(text)
            
            # Enhance with financial keywords
            enhanced_sentiment = self.enhance_sentiment_financial(text, base_sentiment)
            
            sentiment_results.append(enhanced_sentiment)
        
        # Add sentiment columns to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([news_df.reset_index(drop=True), sentiment_df], axis=1)
        
        print("Sentiment analysis completed!")
        print(f"Positive: {(result_df['sentiment_label'] == 'Positive').sum()}")
        print(f"Negative: {(result_df['sentiment_label'] == 'Negative').sum()}")
        print(f"Neutral: {(result_df['sentiment_label'] == 'Neutral').sum()}")
        
        return result_df
    
    def aggregate_daily_sentiment(self, sentiment_df):
        """
        Aggregate sentiment scores by date
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment dataframe with dates
            
        Returns:
            pd.DataFrame: Daily aggregated sentiment
        """
        # Ensure date column is datetime
        if 'date' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Aggregate by date
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'polarity': 'mean',
            'subjectivity': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date', 'sentiment_mean', 'sentiment_std', 'news_count',
            'polarity_mean', 'subjectivity_mean'
        ]
        
        # Fill NaN std with 0
        daily_sentiment['sentiment_std'].fillna(0, inplace=True)
        
        return daily_sentiment
    
    def get_sentiment_summary(self, sentiment_df):
        """
        Get summary statistics of sentiment analysis
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment dataframe
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_articles': len(sentiment_df),
            'positive_count': (sentiment_df['sentiment_label'] == 'Positive').sum(),
            'negative_count': (sentiment_df['sentiment_label'] == 'Negative').sum(),
            'neutral_count': (sentiment_df['sentiment_label'] == 'Neutral').sum(),
            'avg_sentiment_score': sentiment_df['sentiment_score'].mean(),
            'avg_polarity': sentiment_df['polarity'].mean(),
            'avg_subjectivity': sentiment_df['subjectivity'].mean(),
            'sentiment_std': sentiment_df['sentiment_score'].std()
        }
        
        # Calculate percentages
        if summary['total_articles'] > 0:
            summary['positive_pct'] = (summary['positive_count'] / summary['total_articles']) * 100
            summary['negative_pct'] = (summary['negative_count'] / summary['total_articles']) * 100
            summary['neutral_pct'] = (summary['neutral_count'] / summary['total_articles']) * 100
        
        return summary
    
    def save_sentiment_results(self, sentiment_df, filename):
        """
        Save sentiment analysis results to CSV
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment dataframe
            filename (str): Output filename
        """
        filepath = os.path.join(Config.DATA_PROCESSED_PATH, filename)
        sentiment_df.to_csv(filepath, index=False)
        print(f"Sentiment results saved to {filepath}")


if __name__ == "__main__":
    # Test sentiment analysis
    from utils.data_loader import NewsDataLoader
    
    print("Testing Sentiment Analysis...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test with sample texts
    sample_texts = [
        "Apple stock surges to record high on strong earnings beat",
        "Tesla shares plummet amid concerns over production delays",
        "Microsoft announces new cloud services expansion",
        "Amazon faces regulatory challenges in European markets",
        "Google maintains steady growth in advertising revenue"
    ]
    
    print("\nAnalyzing sample headlines:")
    print("-" * 80)
    
    for text in sample_texts:
        sentiment = analyzer.analyze_sentiment_textblob(text)
        enhanced = analyzer.enhance_sentiment_financial(text, sentiment)
        
        print(f"\nHeadline: {text}")
        print(f"Label: {enhanced['sentiment_label']}")
        print(f"Score: {enhanced['sentiment_score']:.3f}")
        print(f"Polarity: {enhanced['polarity']:.3f}")
    
    # Test with real news data
    print("\n" + "="*80)
    print("Testing with news data...")
    
    news_loader = NewsDataLoader()
    news_df = news_loader.fetch_news('AAPL', 'Apple', days_back=7)
    
    if news_df is not None and len(news_df) > 0:
        # Analyze sentiment
        sentiment_df = analyzer.analyze_news_dataframe(news_df)
        
        print("\nSentiment Analysis Results:")
        print(sentiment_df[['headline', 'sentiment_label', 'sentiment_score']].head())
        
        # Get summary
        summary = analyzer.get_sentiment_summary(sentiment_df)
        print("\nSentiment Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
