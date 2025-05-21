# Hybrid Movie Recommendation System

## Objective
Develop a hybrid recommendation system that leverages both *collaborative filtering* (user-item interactions) and *content-based filtering* (item attributes) to suggest movies tailored to individual user preferences.

---

## Features

- *Data Ingestion & Preprocessing*  
  Load and clean the MovieLens 100K dataset, handling missing values and ensuring data consistency.

- *Content-Based Filtering*  
  Extract movie features (e.g., genres) using TF-IDF vectorization and compute similarity via cosine similarity to recommend similar movies.

- *Collaborative Filtering*  
  Use matrix factorization (SVD) to predict user ratings for unseen movies based on past user-item interactions.

- *Hybrid Recommendation Engine*  
  Combine content-based and collaborative filtering outputs via weighted averaging to provide personalized recommendations.

- *Interactive User Interface*  
  Streamlit app allowing users to select a movie they like and specify the number of recommendations to receive hybrid-based suggestions.

- *Evaluation Metrics*  
  Measure model performance with RMSE, MAE, Precision, Recall, and F1-Score.

---

## Dataset

- *MovieLens 100K*  
  Contains 100,000 ratings from 943 users on 1,682 movies, including user ratings, movie metadata (genres, titles), and user information.

---

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/Shahdd-Akram/Hybrid-Movie-Recommendation-System.git
   cd Hybrid-Movie-Recommendation-System