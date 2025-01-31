import pickle
import streamlit as st
import numpy as np
import pandas as pd

# App Header
st.title(" Book Recommendation System")
st.subheader("Find your next favorite book!")

# Load artifacts
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

# Function to fetch book posters
def fetch_poster(suggestion):
    poster_url = []
    for book_id in suggestion:
        book_name = book_pivot.index[book_id]
        # Fetch poster URLs for each recommended book
        matching_books = final_rating[final_rating['Book-Title'] == book_name]
        if not matching_books.empty:
            poster_url.append(matching_books.iloc[0]['Image-URL-L'])
        else:
            # Default poster if URL is missing
            poster_url.append("https://via.placeholder.com/150?text=No+Image")
    return poster_url

# Recommendation function
from sklearn.metrics.pairwise import cosine_similarity

def recommend_book(book_name):
    try:
        # Find the book ID
        book_id = np.where(book_pivot.index == book_name)[0][0]
        
        # Compute similarity with all other books
        similarity_scores = cosine_similarity(book_pivot.iloc[book_id, :].values.reshape(1, -1), book_pivot)
        
        # Get the top recommendations (excluding the queried book)
        top_books = np.argsort(similarity_scores[0])[-7:][::-1]  # Top 6 neighbors
        top_books = [book for book in top_books if book != book_id][:6]  # Exclude queried book
        
        # Fetch book titles and posters
        recommended_books = [book_pivot.index[book] for book in top_books]
        poster_urls = fetch_poster(top_books)
        
        return recommended_books, poster_urls
    except IndexError:
        st.error("Book not found in the dataset. Please try another one.")
        return [], []

# Streamlit UI
selected_book = st.selectbox(" Select or type the name of a book:", book_names)

if st.button(" Show Recommendations"):
    # Generate recommendations
    recommendations, poster_urls = recommend_book(selected_book)
    
    if recommendations:
        st.subheader(f"Books similar to **{selected_book}**:")
        
        # Display recommendations with images in a grid
        cols = st.columns(len(recommendations))
        for i, col in enumerate(cols):
            with col:
                st.image(poster_urls[i], width=150)
                st.caption(recommendations[i])
    else:
        st.warning("No recommendations found. Please try a different book.")
