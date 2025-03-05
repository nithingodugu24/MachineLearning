from flask import Flask, render_template, jsonify
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the processed movie dataset and model
movies_df = pd.read_pickle(
    "processed_movies.pkl"
)  # Ensure your dataset is stored as a pickle file
from scipy.spatial.distance import euclidean, hamming


def weighted_distance(movie1, movie2):
    movie1 = movie1.ravel()  # Ensure it's a 1D array
    movie2 = movie2.ravel()  # Ensure it's a 1D array

    actors_dist = euclidean(movie1[4:6], movie2[4:6]) * 1.6
    years_dist = euclidean([movie1[3]], [movie2[3]]) * 0.03
    rating_dist = euclidean([movie1[2]], [movie2[2]]) * 0.45
    type_dist = hamming([movie1[6]], [movie2[6]]) * 1.05
    gen_dist = hamming(movie1[7:], movie2[7:]) * 0.45

    return actors_dist + years_dist + rating_dist + type_dist + gen_dist


knn_model = pickle.load(open("knn_model.pkl", "rb"))


# Fetch movie recommendations
def get_recommendations(movie_id):
    index = movies_df[movies_df["id"] == movie_id].index[0]
    distances, indices = knn_model.kneighbors([movies_df.iloc[index, 2:].to_numpy()])
    recommended_indices = indices[0][1:]
    return movies_df.iloc[recommended_indices][["id", "title"]].to_dict(
        orient="records"
    )


@app.route("/")
def index():
    sections = {
        "Top Rated": movies_df.sort_values("rating", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "Action Movies": movies_df[movies_df["Action"] == 1]
        .head(10)
        .to_dict(orient="records"),
        "Comedy Movies": movies_df[movies_df["Comedy"] == 1]
        .head(10)
        .to_dict(orient="records"),
    }
    return render_template("index.html", sections=sections)


@app.route("/movie/<imdb_id>")
def movie_details(imdb_id):
    movie = movies_df[movies_df["id"] == imdb_id].to_dict(orient="records")[0]
    recommendations = get_recommendations(imdb_id)
    return render_template("movie.html", movie=movie, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
