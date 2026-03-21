"""
Data Loader Module
MSc IT Project - Stock Price Forecasting

This module handles fetching stock data from Yahoo Finance
and news data from News API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
import os
import sys
from urllib.parse import quote as url_quote

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.symbol_catalog import get_company_name_for_symbol, get_market_data_symbol


class StockDataLoader:
    """Fetch and process stock market data"""
    
    def __init__(self, symbol=None):
        """
        Initialize StockDataLoader
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
        """
        self.symbol = symbol or Config.DEFAULT_STOCK_SYMBOL
        self.market_symbol = get_market_data_symbol(self.symbol)
        
    def fetch_historical_data(self, start_date=None, end_date=None):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        start_date = start_date or Config.START_DATE
        end_date = end_date or Config.END_DATE

        print(f"Fetching data for {self.symbol} ({self.market_symbol}) from {start_date} to {end_date}...")

        try:
            # Download data using yfinance first
            stock = yf.Ticker(self.market_symbol)
            df = stock.history(start=start_date, end=end_date)

            if df is None or df.empty:
                raise ValueError("yfinance returned no data")

            df = self._finalize_history_dataframe(df)

            print(f"Successfully fetched {len(df)} records for {self.symbol}")
            return df

        except Exception as e:
            print(f"yfinance failed: {str(e)}")
            print("Falling back to Yahoo Finance chart API...")

            df = self._fetch_from_yahoo_chart(start_date, end_date)
            if df is None or df.empty:
                print(f"No data found for {self.symbol} via fallback.")
                return None

            print(f"Successfully fetched {len(df)} records for {self.symbol} via fallback.")
            return df

    def _fetch_from_yahoo_chart(self, start_date, end_date):
        """
        Fallback: fetch price history using Yahoo Finance chart API directly.
        """
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(days=1)

            period1 = int(start_dt.timestamp())
            period2 = int(end_dt.timestamp())

            yahoo_symbol = url_quote(self.market_symbol, safe='')
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                "interval": "1d",
                "period1": period1,
                "period2": period2,
                "events": "history"
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            payload = response.json()
            chart_error = payload.get("chart", {}).get("error")
            if chart_error:
                raise ValueError(chart_error.get("description") or "Yahoo chart API returned an error")

            result = payload.get("chart", {}).get("result")
            if not result:
                return None

            result = result[0]
            timestamps = result.get("timestamp", [])
            indicators = result.get("indicators", {})
            quote_data = indicators.get("quote", [{}])[0]
            adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")

            if not timestamps:
                # Retry with range-based request (server chooses end date)
                params = {
                    "interval": "1d",
                    "range": "2y",
                    "events": "history"
                }
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()

                payload = response.json()
                chart_error = payload.get("chart", {}).get("error")
                if chart_error:
                    raise ValueError(chart_error.get("description") or "Yahoo chart API returned an error")

                result = payload.get("chart", {}).get("result")
                if not result:
                    return None

                result = result[0]
                timestamps = result.get("timestamp", [])
                indicators = result.get("indicators", {})
                quote_data = indicators.get("quote", [{}])[0]
                adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")

            if not timestamps:
                return None

            df = pd.DataFrame({
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Open": quote_data.get("open"),
                "High": quote_data.get("high"),
                "Low": quote_data.get("low"),
                "Close": quote_data.get("close"),
                "Volume": quote_data.get("volume")
            })

            if adjclose:
                df["Adj_Close"] = adjclose
            else:
                df["Adj_Close"] = df["Close"]

            df = self._finalize_history_dataframe(df)
            return df

        except Exception as e:
            print(f"Fallback Yahoo chart error: {str(e)}")
            return None

    def _finalize_history_dataframe(self, df):
        """Normalize Yahoo history output and keep one row per trading date."""
        if df is None or df.empty:
            return df

        df = df.copy()

        if 'Date' not in df.columns:
            df.reset_index(inplace=True)

        if 'Date' not in df.columns:
            raise ValueError("Historical data is missing the 'Date' column.")

        if 'Adj Close' in df.columns:
            df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
        elif 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        try:
            df['Date'] = df['Date'].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        df['Date'] = df['Date'].dt.normalize()

        expected = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
        df = df[[col for col in expected if col in df.columns]]
        df = df.dropna(subset=['Date', 'Close'])
        df = df.sort_values('Date')

        duplicate_count = int(df.duplicated(subset=['Date']).sum())
        if duplicate_count:
            print(f"Dropping {duplicate_count} duplicate history rows for {self.symbol}")
            df = df.drop_duplicates(subset=['Date'], keep='last')

        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

        return df.reset_index(drop=True)
    
    def save_to_csv(self, df, filename=None):
        """
        Save dataframe to CSV file
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        try:
            if filename is None:
                filename = f"{self.symbol}_historical_data.csv"
            
            filepath = os.path.join(Config.DATA_RAW_PATH, filename)
            df.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
    
    def load_from_csv(self, filename=None):
        """
        Load data from CSV file
        
        Args:
            filename (str): Input filename
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if filename is None:
                filename = f"{self.symbol}_historical_data.csv"
            
            filepath = os.path.join(Config.DATA_RAW_PATH, filename)
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            
            print(f"Data loaded from {filepath}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None


class NewsDataLoader:
    """Fetch news articles for sentiment analysis"""
    
    def __init__(self, api_key=None):
        """
        Initialize NewsDataLoader
        
        Args:
            api_key (str): News API key
        """
        self.api_key = api_key or Config.NEWS_API_KEY
        
        # Initialize News API client only if valid key provided
        if self.api_key and self.api_key != 'YOUR_NEWS_API_KEY_HERE':
            try:
                self.newsapi = NewsApiClient(api_key=self.api_key)
            except:
                self.newsapi = None
                print("Warning: Invalid News API key. Using fallback method.")
        else:
            self.newsapi = None
            print("Warning: No News API key provided. Using fallback method.")
    
    def fetch_news(self, symbol, company_name=None, days_back=7):
        """
        Fetch news articles related to a stock
        
        Args:
            symbol (str): Stock ticker symbol
            company_name (str): Full company name
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: News articles data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Try News API first
            if self.newsapi:
                return self._fetch_from_newsapi(symbol, company_name, start_date, end_date)
            else:
                # Fallback: Generate sample news data
                return self._generate_sample_news(symbol, days_back)
                
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return self._generate_sample_news(symbol, days_back)
    
    def _fetch_from_newsapi(self, symbol, company_name, start_date, end_date):
        """Fetch from News API"""
        try:
            # Determine search query
            if company_name:
                query = f"{company_name} OR {symbol}"
            else:
                query = symbol
            
            # Fetch articles
            articles = self.newsapi.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=Config.MAX_NEWS_ARTICLES
            )
            
            # Process articles
            news_list = []
            for article in articles.get('articles', []):
                news_list.append({
                    'date': article['publishedAt'][:10],
                    'headline': article['title'],
                    'description': article['description'] or '',
                    'source': article['source']['name'],
                    'url': article['url']
                })
            
            df = pd.DataFrame(news_list)
            df['date'] = pd.to_datetime(df['date'])
            
            print(f"Fetched {len(df)} news articles for {symbol}")
            return df
            
        except Exception as e:
            print(f"News API error: {str(e)}")
            return self._generate_sample_news(symbol, 7)
    
    def _generate_sample_news(self, symbol, days_back=7):
        """
        Generate sample news data for demonstration
        This is used when News API is unavailable
        """
        print(f"Generating sample news data for {symbol}...")
        
        # Sample headlines with varying sentiment
        positive_headlines = [
            f"{symbol} stock surges on strong earnings report",
            f"{symbol} announces breakthrough product innovation",
            f"Analysts upgrade {symbol} rating to 'Strong Buy'",
            f"{symbol} beats quarterly revenue expectations",
            f"{symbol} expands market share in key segment"
        ]
        
        negative_headlines = [
            f"{symbol} faces regulatory challenges",
            f"{symbol} misses earnings estimates",
            f"Concerns grow over {symbol}'s market position",
            f"{symbol} announces workforce reduction",
            f"Analysts downgrade {symbol} amid market concerns"
        ]
        
        neutral_headlines = [
            f"{symbol} holds annual shareholders meeting",
            f"{symbol} maintains steady market position",
            f"Market watches {symbol} quarterly results",
            f"{symbol} announces routine board changes",
            f"Industry report mentions {symbol} among peers"
        ]
        
        news_list = []
        end_date = datetime.now()
        
        for i in range(days_back):
            date = end_date - timedelta(days=i)
            
            # Randomly select headlines
            import random
            sentiment_type = random.choice(['positive', 'negative', 'neutral'])
            
            if sentiment_type == 'positive':
                headline = random.choice(positive_headlines)
            elif sentiment_type == 'negative':
                headline = random.choice(negative_headlines)
            else:
                headline = random.choice(neutral_headlines)
            
            news_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'headline': headline,
                'description': headline,
                'source': random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'Financial Times']),
                'url': f'https://example.com/news/{i}'
            })
        
        df = pd.DataFrame(news_list)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Generated {len(df)} sample news articles")
        return df


def get_company_name(symbol):
    """
    Get company name from stock symbol
    
    Args:
        symbol (str): Stock ticker symbol
        
    Returns:
        str: Company name
    """
    return get_company_name_for_symbol(symbol)


if __name__ == "__main__":
    # Test the data loader
    print("Testing Stock Data Loader...")
    
    loader = StockDataLoader('AAPL')
    df = loader.fetch_historical_data()
    
    if df is not None:
        print("\nFirst few records:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        # Save to CSV
        loader.save_to_csv(df)
    
    # Test news loader
    print("\n" + "="*50)
    print("Testing News Data Loader...")
    
    news_loader = NewsDataLoader()
    news_df = news_loader.fetch_news('AAPL', 'Apple', days_back=5)
    
    if news_df is not None:
        print("\nFirst few news articles:")
        print(news_df.head())
