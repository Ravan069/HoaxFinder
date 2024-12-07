from GoogleNews import GoogleNews
import streamlit as st
from sentence_transformers import CrossEncoder

# Page configuration
st.set_page_config(page_title="HoaxFinder", page_icon="🕵️", layout="wide")
st.header("🕵️ HoaxFinder")

# Initialize model and GoogleNews
model = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v4')
googlenews = GoogleNews(lang='en', region='US')

# Fetch news function
def fetch_news(query):
    googlenews.search(query)
    return googlenews.get_texts()

# Validation function
def valid(query, news):
    # Prepare query-news pairs
    pairs = [(query, news_item) for news_item in news]
    
    # Predict similarity scores
    scores = model.predict(pairs, show_progress_bar=False)
    
    # Calculate average score
    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0

    
    # Display news and results
    st.write("## 📰 According to Google News:")
    for i, item in enumerate(news):
        st.write(f"**News {i+1}:** {item}")
    
    # Determine if the news is a hoax
    if avg_score >= 0.8:
        st.write("### 🕵️ says the news is a **Hoax!** 🛑")
    else:
        st.write("### 🕵️ says the news is **Not a Hoax.** ✅")
    
    # Display the average similarity score
    st.write(f"**Average Similarity Score:** {avg_score:.2f}")

# Main application
if __name__ == "__main__":
    # Input from the user
    if query := st.text_input("Enter the news title"):
        # Fetch news from Google News
        google_news = fetch_news(query)
        
        if google_news:
            # Validate the news
            valid(query, google_news)
        else:
            st.write("No news articles found for the given query. Try another search.")
