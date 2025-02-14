import json
import time
import requests
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import instaloader  # Instagram Scraper
import facebook_scraper  # Facebook Scraper
import linkedin_api  # LinkedIn API
import TikTokApi  # TikTok Scraper
import streamlit as st  # Dashboard
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    return "Neutral"

# Function to compute embeddings
def compute_embedding(text):
    return bert_model.encode(text, convert_to_numpy=True)

# Function to fetch Facebook data
def fetch_facebook_data(page_name, max_posts=50):
    try:
        data = []
        for post in facebook_scraper.get_posts(page_name, pages=5):
            text = post.get('text', '')
            data.append({
                "platform": "Facebook",
                "content": text,
                "likes": post.get('likes', 0),
                "retweets": None,
                "created_at": post.get('time', datetime.now()),
                "sentiment": analyze_sentiment(text),
                "embedding": compute_embedding(text)
            })
            if len(data) >= max_posts:
                break
        return data
    except Exception as e:
        logging.error(f"Error fetching Facebook data: {e}")
        return []

# Function to fetch Instagram data
def fetch_instagram_data(username, max_posts=50):
    try:
        loader = instaloader.Instaloader()
        profile = instaloader.Profile.from_username(loader.context, username)
        data = []
        for post in profile.get_posts():
            text = post.caption or ""
            data.append({
                "platform": "Instagram",
                "content": text,
                "likes": post.likes,
                "retweets": None,
                "created_at": post.date,
                "sentiment": analyze_sentiment(text),
                "embedding": compute_embedding(text)
            })
            if len(data) >= max_posts:
                break
        return data
    except Exception as e:
        logging.error(f"Error fetching Instagram data: {e}")
        return []

# Function to fetch LinkedIn data
def fetch_linkedin_data(company_name, max_posts=50):
    logging.info("Fetching LinkedIn data is currently not implemented.")
    return []

# Function to fetch TikTok data
def fetch_tiktok_data(hashtag, max_posts=50):
    logging.info("Fetching TikTok data is currently not implemented.")
    return []

# Function to display dashboard
def display_dashboard():
    st.title("Social Media Analysis Dashboard")
    facebook_page = st.sidebar.text_input("Enter Facebook Page Name", "your_page")
    instagram_username = st.sidebar.text_input("Enter Instagram Username", "your_instagram")
    linkedin_company = st.sidebar.text_input("Enter LinkedIn Company Name", "your_company")
    tiktok_hashtag = st.sidebar.text_input("Enter TikTok Hashtag", "your_hashtag")
    if st.sidebar.button("Run Analysis"):
        facebook_data = fetch_facebook_data(facebook_page)
        instagram_data = fetch_instagram_data(instagram_username)
        linkedin_data = fetch_linkedin_data(linkedin_company)
        tiktok_data = fetch_tiktok_data(tiktok_hashtag)
        all_data = facebook_data + instagram_data + linkedin_data + tiktok_data
        if all_data:
            df = pd.DataFrame(all_data)
            st.dataframe(df)
        else:
            st.write("No data collected.")

# Main function to initialize Streamlit
def main():
    st.set_page_config(layout="wide")
    display_dashboard()

if __name__ == "__main__":
    main()
