from GoogleNews import GoogleNews
import streamlit as st
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="HoaxFinder", page_icon="🕵️", layout="wide")
st.header("🕵️ HoaxFinder ")

model = CrossEncoder('abbasgolestani/ag-nli-DeTS-sentence-similarity-v4')
googlenews = GoogleNews()

def fetch_news(query):
    googlenews.search(query)
    return googlenews.get_texts()

def valid(query, news):   
    pairs = zip(query, news)
    list_pairs=list(pairs)
    scores1 = model.predict(list_pairs, show_progress_bar=False)
    
    result = 0 
    for score in scores1:
        sum+=score
        result = sum/len(scores1)
        
    print_news = str(news)    
         
    if result >= 0.8 :
        st.write("""
                ### 🕵️ says the news is Hoax!!! 
                 
                ## 📰 According to Google News:
                 
                 """, print_news[2:-2],  sum)
    else :
        st.write("""
                ### 🕵️ says the news is Not a Hoax!!! 
                 
                ## 📰 According to Google News:
                 
                 """, print_news[2:-2],  sum)   
              

if __name__ == "__main__":
    
    if query := st.text_input("Enter the news title"):
        google_news = fetch_news(query)
        valid(query, google_news)