# Data preprocessing, since the data downloaded is in .dat format
# Import packages
from utilities import *
import re

# define relative path for the files
relative_path = "./parsed data/"
user_data_file = 'users.csv'
movie_data_file = 'movies.csv'
rating_data_file = 'ratings.csv'

# Reading ratings file
# Ignore the timestamp column
ratings = pd.read_csv(relative_path + rating_data_file, sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id',
                                                                                               'rating'])

# Reading users file
users = pd.read_csv(relative_path + user_data_file, sep='\t', encoding='latin-1', usecols=['user_id', 'gender',
                                                                                           'zipcode', 'age_desc',
                                                                                           'occ_desc'])

# Reading movies file
movies = pd.read_csv(relative_path + movie_data_file, sep='\t', encoding='latin-1', usecols=['movie_id', 'title',
                                                                                             'genres'])
# creating list with unique genres
genres = list(set('|'.join(list(movies["genres"].unique())).split('|')))

# Creating dummy columns for each genre
for genre in genres:
    movies[genre] = movies['genres'].map(lambda val: 1 if genre in val else 0)

# Creating column with film year
movies['year'] = movies['title'].map(lambda val: int(re.search('\(([0-9]{4})\)', val).group(1))
                                     if re.search('\(([0-9]{4})\)', val) != None
                                     else 0)
# Film Decade
for decade in range(1930, 2020, 10):
    movies['decade_' + str(decade)] = np.where((movies['year'] < decade + 10) & (movies['year'] >= decade), 1, 0)
#     print('column created','decade_' + str(decade))

movies['decade_none'] = np.where(movies['year'] == 0, 1, 0)
movies['decade_other'] = np.where((movies['year'] != 0) & (movies['year'] < 1930), 1, 0)


# Droping genres
movies.drop('genres', axis=1,inplace= True)
df = pd.merge(ratings, movies, on='movie_id')
print(df.shape)

categories = genres
films_rated = movies.to_dict()

# film_name = 'Inception (2010)'
film_name ='Toy Story (1995)'
user_id = 611

generate_recomendations(df, film_name, films_rated, 5)



# movies = movies.join(movies.genres.str.get_dummies("|"))
#
# # compute the cosine similarity
# cos_sim = cosine_similarity(movies.iloc[:,3:])
#
#
#
# # Fill NaN values in user_id and movie_id column with 0
# ratings['user_id'] = ratings['user_id'].fillna(0)
# ratings['movie_id'] = ratings['movie_id'].fillna(0)
#
# # Replace NaN values in rating column with average of all values
# ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())
#
# # Break up the big genre string into a string array
# movies['genres'] = movies['genres'].str.split('|')
# # Convert genres to string value
# movies['genres'] = movies['genres'].fillna("").astype('str')
#
#
# # Fill NaN values in user_id and movie_id column with 0
# ratings['user_id'] = ratings['user_id'].fillna(0)
# ratings['movie_id'] = ratings['movie_id'].fillna(0)
#
# # Replace NaN values in rating column with average of all values
# ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())
#
# # Randomly sample 1% of the ratings dataset
# small_data = ratings.sample(frac=0.02)
# # Check the sample info
# print(small_data.info())
#
# train_data, test_data = train_test_split(small_data, test_size=0.2)
#
# # Create two user-item matrices, one for training and another for testing
# train_data_matrix = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
# test_data_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
#
# # Check their shape
# print(train_data_matrix.shape)
# print(test_data_matrix.shape)
#
# # Here we are using pearson similarity, we can use cosine similarity as well
# # User Similarity Matrix
# user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
# user_correlation[np.isnan(user_correlation)] = 0
# print(user_correlation[:4, :4])
#
# # Item Similarity Matrix
# item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
# item_correlation[np.isnan(item_correlation)] = 0
# print(item_correlation[:4, :4])
#
#
# # Predict ratings on the training data with both similarity score
# user_prediction = predict(train_data_matrix, user_correlation, type='user')
# item_prediction = predict(train_data_matrix, item_correlation, type='item')
#
# # RMSE on the test data
# print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
# print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
#
# # RMSE on the train data
# print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
# print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))
#
#
# # Cosine similarity option
# movies = movies.join(movies.genres.str.get_dummies("|"))
