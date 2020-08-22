from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Reading ratings file
    user_data_file = 'users.csv'
    movie_data_file = 'movies.csv'
    rating_data_file = 'ratings.csv'

    # Ignore the timestamp column
    ratings = pd.read_csv(rating_data_file, sep='\t', encoding='latin-1',
                          usecols=['user_id', 'movie_id',
                                   'rating'])

    # Reading users file
    users = pd.read_csv(user_data_file, sep='\t', encoding='latin-1', usecols=['user_id', 'gender',
                                                                                               'zipcode', 'age_desc',
                                                                                               'occ_desc'])

    # Reading movies file
    movies = pd.read_csv(movie_data_file, sep='\t', encoding='latin-1', usecols=['movie_id', 'title',
                                                                                                 'genres'])

    # Create a wordcloud of the movie titles
    movies['title'] = movies['title'].fillna("").astype('str')
    title_corpus = ' '.join(movies['title'])

    # Join all 3 files into one dataframe
    dataset = pd.merge(pd.merge(movies, ratings), users)

    # Display 20 movies with highest ratings
    dataset[['title', 'genres', 'rating']].sort_values('rating', ascending=False).head(20)

    # Make a census of the genre keywords
    genre_labels = set()
    for s in movies['genres'].str.split('|').values:
        genre_labels = genre_labels.union(set(s))

    # Function that counts the number of times each of the genre keywords appear
    def count_word(dataset, ref_col, census):
        # create a dictionary to store the key and values
        keyword_count = dict()
        for s in census:
            keyword_count[s] = 0
        for census_keywords in dataset[ref_col].str.split('|'):
            if type(census_keywords) == float and pd.isnull(census_keywords):
                continue
            for s in [s for s in census_keywords if s in census]:
                if pd.notnull(s):
                    keyword_count[s] += 1
        # convert the dictionary in a list to sort the keywords by frequency
        keyword_occurences = []
        for k, v in keyword_count.items():
            keyword_occurences.append([k, v])
        keyword_occurences.sort(key=lambda x: x[1], reverse=True)
        return keyword_occurences, keyword_count

    # Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
    keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
    keyword_occurences[:5]

    # Break up the big genre string into a string array
    movies['genres'] = movies['genres'].str.split('|')
    # Convert genres to string value
    movies['genres'] = movies['genres'].fillna("").astype('str')

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(movies['genres'])
    tfidf_matrix.shape

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim[:4, :4]

    # Build a 1-dimensional array with movie titles
    titles = movies['title']
    indices = pd.Series(movies.index, index=movies['title'])

    # Function that get movie recommendations based on the cosine similarity score of movie genres
    def genre_recommendations(title):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]

    # Alternative Usage of Saved Model
    # joblib.dump(genre_recommendations, 'genre_recommender_model.pkl')
    # recomm_model = open('genre_recommender_model.pkl', 'rb')
    # clf = joblib.load(recomm_model)

    # if request.method == 'POST':
    #     message = request.form['message']
    #     data = [message]
    #     my_prediction = genre_recommendations(data).head(20)
    message = request.form['message']
    data = [message]
    prediction_text = genre_recommendations(data).head(20)
    prediction_text = np.array(prediction_text)
    return render_template('home.html', prediction_text=prediction_text)
    # return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=False)

