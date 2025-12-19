"""
Script to generate similarity.pkl from movie_dict.pkl
This recreates the similarity matrix using cosine similarity on movie tags.
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading movie data...")
movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

print(f"Loaded {len(movies)} movies")
print(f"Columns: {list(movies.columns)}")

# Extract tags column for similarity computation
print("Creating feature vectors from tags...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags'].values.astype('U')).toarray()

print(f"Vector shape: {vector.shape}")

# Compute cosine similarity
print("Computing cosine similarity matrix...")
similarity = cosine_similarity(vector)

print(f"Similarity matrix shape: {similarity.shape}")

# Save the similarity matrix
print("Saving similarity.pkl...")
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

print("Done! similarity.pkl has been generated successfully.")

