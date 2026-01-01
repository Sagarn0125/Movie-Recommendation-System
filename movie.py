import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('movies.csv')
    return data

# Preprocess the data
def preprocess_data(data):
    selected_features = ['genres','keywords','tagline','cast','director']
    for feature in selected_features:
        data[feature] = data[feature].fillna('')
    combined_features = data['genres'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast'] + ' ' + data['director']
    return combined_features

# Generate feature vectors
def generate_similarity_matrix(combined_features):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

# Recommendation function
def recommend_movies(movie_name, movies_data, similarity):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return []

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_titles = []
    for i, movie in enumerate(sorted_similar_movies):
        if i == 0:
            continue  # skip the input movie itself
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_titles.append(title_from_index)
        if len(recommended_titles) >= 30:
            break
    return recommended_titles

# ----------------- STREAMLIT UI -----------------

st.title("🎬 Movie Recommendation System")
st.write("Get movie suggestions based on your favorite film!")

# Load and prepare data
movies_data = load_data()

# Make sure there is an 'index' column
if 'index' not in movies_data.columns:
    movies_data.reset_index(inplace=True)

combined_features = preprocess_data(movies_data)
similarity = generate_similarity_matrix(combined_features)

# User Input
movie_name = st.text_input("Enter your favourite movie name")

if movie_name:
    recommendations = recommend_movies(movie_name, movies_data, similarity)
    
    if recommendations:
        st.subheader("Movies suggested for you:")
        for idx, title in enumerate(recommendations, start=1):
            st.write(f"{idx}. {title}")
    else:
        st.warning("Sorry, no close match found for the entered movie.")