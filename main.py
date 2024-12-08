from GoogleNews import GoogleNews
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="HoaxFinder", page_icon="🕵️", layout="wide")
st.header("🕵️ HoaxFinder ")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Use Sentence-BERT
googlenews = GoogleNews()

def fetch_news(query):
    googlenews.search(query)
    return googlenews.get_texts()

def get_similarity_score(query, news):
    embeddings1 = model.encode(query, convert_to_tensor=True)
    embeddings2 = model.encode(news, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores[0][0].item()

def valid(query, news):
    if not news:
        st.warning("No news found for the given query.")
        return

    scores = [get_similarity_score(query, item) for item in news]
    max_score = max(scores)
    confidence = (max_score + 2) / 4  # Normalize score to a percentage (0-1)

    if confidence > 0.6:  # Adjust threshold as needed
        st.write(f"""
            ### 🕵️ says the news might be a Hoax! 
            Confidence: {confidence:.2f}

            ## 📰 According to Google News:
            """)
    else:
        st.write(f"""
            ### 🕵️ says the news is likely Not a Hoax! 
            Confidence: {confidence:.2f}

            ## 📰 According to Google News:
            """)

    for news_item in news:
        st.write(f"- {news_item}")

if __name__ == "__main__":
    if query := st.text_input("Enter the news title"):
        google_news = fetch_news(query)
        valid(query, google_news)