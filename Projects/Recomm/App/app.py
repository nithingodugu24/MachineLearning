from flask import Flask, render_template, jsonify
import pandas as pd
import pickle, random
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the processed movie dataset and model
df = pd.read_pickle("models/v1/recomm-data.pkl")
details_df = pd.read_pickle("models/v1/imdb-details.pkl")

from scipy.spatial.distance import euclidean, hamming


from scipy.spatial.distance import euclidean, hamming


def weighted_distance(movie1, movie2, maxtrix=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
    movie1 = movie1.flatten()  # Ensure it's 1D
    movie2 = movie2.flatten()  # Ensure it's 1D

    voteCount = euclidean([movie1[0]], [movie2[0]]) * 3.0 * maxtrix[0]
    country = euclidean([movie1[1]], [movie2[1]]) * 2.5 * maxtrix[1]
    language = euclidean([movie1[2]], [movie2[2]]) * 4.0 * maxtrix[2]

    kw = euclidean(movie1[3:6], movie2[3:6]) * 3.5 * maxtrix[3]
    cast = euclidean(movie1[6:8], movie2[6:8]) * 3.0 * maxtrix[4]
    crew = euclidean(movie1[8:10], movie2[8:10]) * 2.0 * maxtrix[5]

    rating = euclidean([movie1[10]], [movie2[10]]) * 5.0 * maxtrix[6]
    year = euclidean([movie1[11]], [movie2[11]]) * 4.5 * maxtrix[7]
    mtype = hamming([movie1[12]], [movie2[12]]) * 1.5 * maxtrix[8]

    genres = hamming(movie1[13:], movie2[13:]) * 6.0 * maxtrix[9]

    return float(
        voteCount
        + country
        + language
        + kw
        + cast
        + crew
        + rating
        + year
        + mtype
        + genres
    )


knn_model = pickle.load(open("models/v1/refinedv1-imdb-recomm.pkl", "rb"))


# Fetch movie recommendations
import numpy as np


def get_recommendations(movie_id, k=5, user_prefs=None):
    index = df[df["id"] == movie_id].index[0]
    target_movie = df.iloc[index, 1:].to_numpy()

    # Default weight matrix (all features equally weighted)
    weight_matrix = np.ones(10)

    # Modify weights based on user preferences
    if user_prefs:
        if user_prefs.get("prefer_recent"):
            weight_matrix[7] = 5  # Increase weight for release year
        if user_prefs.get("prefer_high_rated"):
            weight_matrix[6] = 6  # Increase weight for ratings
        if user_prefs.get("prefer_action_movies"):
            weight_matrix[9] = 7  # Increase weight for genres
        if user_prefs.get("prefer_cast_crew"):
            weight_matrix[4] = 30  # Increase cast importance
            weight_matrix[5] = 10  # Increase crew importance

    # Compute distances manually for all movies
    distances = []
    for i, row in df.iterrows():
        if i == index:  # Skip the input movie itself
            continue
        other_movie = row.iloc[1:].to_numpy()
        dist = weighted_distance(target_movie, other_movie, weight_matrix)
        distances.append((dist, i))

    # Get the top 10 nearest neighbors (smallest distances)
    distances.sort()
    nearest_indices = [i[1] for i in distances[:k]]

    return details_df.iloc[nearest_indices][["id", "title", "imageUrl"]].to_dict(
        orient="records"
    )


@app.route("/")
def index():
    sections = {
        "Top Rated": details_df.sort_values("year", ascending=False)
        .head(1000)
        .sort_values("voteCount")
        .sort_values("rating", ascending=False)
        .sample(10)
        .to_dict(orient="records"),
        "Indian": details_df[details_df["country"] == "IN"]
        .sort_values("rating", ascending=False)
        .sort_values("voteCount")
        .sample(10)
        .to_dict(orient="records"),
        "Action Movies": details_df[df["Action"] == 1]
        .sort_values("rating", ascending=False)
        .sample(10)
        .to_dict(orient="records"),
        "Comedy Movies": details_df[df["Comedy"] == 1]
        .sort_values("rating", ascending=False)
        .sample(10)
        .to_dict(orient="records"),
    }
    for key, val in sections.items():
        for movie in val:
            movie["imageUrl"] = movie["imageUrl"].replace("_V1_", "_V0_UY248_")

    return render_template("index.html", sections=sections)


@app.route("/movie/<imdb_id>")
def movie_details(imdb_id):
    movie = details_df[df["id"] == imdb_id].to_dict(orient="records")[0]
    movie["imageUrl"] = movie["imageUrl"].replace("_V1_", "_V0_UY720_")
    recommendations = random.sample(get_recommendations(imdb_id, k=20), 5)
    for recom in recommendations:
        recom["imageUrl"] = recom["imageUrl"].replace("_V1_", "_V0_UY248_")
    return render_template("movie.html", movie=movie, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
