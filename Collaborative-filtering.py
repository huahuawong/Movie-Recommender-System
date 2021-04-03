# Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import pairwise_distances
import math

# define relative path for the files
relative_path = "./parsed data/"
user_data_file = 'users.csv'
movie_data_file = 'movies.csv'
rating_data_file = 'ratings.csv'

# Reading ratings file, ignore the timestamp column
ratings = pd.read_csv(relative_path + rating_data_file, sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id',
                                                                                               'rating'])

# Reading users file
users = pd.read_csv(relative_path + user_data_file, sep='\t', encoding='latin-1', usecols=['user_id', 'gender',
                                                                                           'zipcode', 'age_desc',
                                                                                           'occ_desc'])

# Reading movies file
movies = pd.read_csv(relative_path + movie_data_file, sep='\t', encoding='latin-1', usecols=['movie_id', 'title',
                                                                                             'genres'])

# Inspect files columns
ratings.head(5)
users.head(5)
movies.head(5)

# ratings.rating.plot.hist(bins=10, color = "skyblue", lw=0)
# plt.title("Distribution of Users' Ratings")
# plt.ylabel('Number of Ratings')
# plt.xlabel('Rating (Out of 5)')
# plt.show()

# Check for null values
print("Count of null values in ratings:\n", ratings.isnull().sum())
print("Count of null values in users:\n", users.isnull().sum())
print("Count of null values in movies:\n", movies.isnull().sum())

# Seems like there are not any null values
#  Randomly sample % of the ratings dataset
small_data = ratings.sample(frac=0.1)

# Check the sample info
print(small_data.info())

train_data, test_data = train_test_split(small_data, test_size=0.2)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = train_data[['user_id', 'movie_id', 'rating']]
test_data_matrix = test_data[['user_id', 'movie_id', 'rating']]

# Check their shape
print(train_data_matrix.shape); print(test_data_matrix.shape)

# Collaborative filtering is based on the assumption that people like things that are similar to the items that they
# like or stuff that other people with similar taste like as well. For now, we'll look into memory-based CF model, which
# are item based and user based
n_users_train = train_data_matrix['user_id'].unique().max()
n_items_train = train_data_matrix['movie_id'].unique().max()

n_users_test = test_data_matrix['user_id'].unique().max()
n_items_test = test_data_matrix['movie_id'].unique().max()

train_matrix = np.zeros((n_users_train, n_items_train))
for line in train_data_matrix.itertuples():
    train_matrix[line[1]-1,line[2]-1] = line[3]

test_matrix = np.zeros((n_users_test, n_items_test))
for line in test_data_matrix.itertuples():
    test_matrix[line[1]-1,line[2]-1] = line[3]


def predict(train_matrix, user_similarity, type='user', n_similar=20):
    if type == 'user':
        similar_n = user_similarity.argsort()[:,-n_similar:][:,::-1]
        pred = np.zeros((n_users_train,n_items_train))
        for i,users in enumerate(similar_n):
            similar_users_indexes = users
            similarity_n = user_similarity[i,similar_users_indexes]
            matrix_n = train_matrix[similar_users_indexes,:]
            rated_items = similarity_n[:,np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:,np.newaxis])/ similarity_n.sum()
            pred[i,:]  = rated_items
    elif type == 'item':
        similar_n = item_similarity.argsort()[:,-n_similar:][:,::-1]
        print('similar_n shape: ', similar_n.shape)
        pred = np.zeros((n_users_train,n_items_train))

        for i,items in enumerate(similar_n):
            similar_items_indexes = items
            similarity_n = item_similarity[i,similar_items_indexes]
            matrix_n = train_matrix[:,similar_items_indexes]
            rated_items = matrix_n.dot(similarity_n)/similarity_n.sum()
            pred[:,i]  = rated_items
    return pred


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

# train_matrix.shape, test_matrix.shape
user_similarity = pairwise_distances(train_matrix, metric='cosine')
print('shape: ',user_similarity.shape)

user_predictions = predict(train_matrix,user_similarity, type='user', n_similar=50) + train_matrix.mean(axis=1)[:, np.newaxis]
print('predictions shape ',user_predictions.shape)
print('User-based CF RMSE: ' ,math.sqrt(mean_squared_error(user_predictions,test_matrix)))

# Item-item
item_similarity = pairwise_distances(train_matrix.T, metric = 'cosine')
# item_similarity.shape

item_predictions = predict(train_matrix,item_similarity, type='user', n_similar=50)
print('predictions shape ',item_predictions.shape)
print('Item-based CF RMSE: ' ,math.sqrt(mean_squared_error(item_predictions,test_matrix)))

# Generate recommendations for users
def generate_recom(user_id, train_matrix, type='user'):
    if type == 'user':
        predictions = user_predictions
    else:
        predictions = item_predictions
    user_ratings = predictions[user_id-1,:]
    train_unkown_indices = np.where(train_matrix[user_id-1,:] == 0)[0]
    user_recommendations = user_ratings[train_unkown_indices]
    print('\nRecommendations for user {} based on ' + str(type) + '-based CF are the movies: '.format(user_id))
    for movie_id in user_recommendations.argsort()[-5:][: : -1]:
        print(np.array(movies.loc[movies['movie_id'] == movie_id+1].title))
        # print(movie_id +1)
    return

generate_recom(user_id=886, train_matrix=test_matrix, type='user')

## Model-based collaborative filtering
# Previously we've tried item-item and user-user. These belong to memory based CF model, now let's try one of the model
# based CF model
from scipy.sparse.linalg import svds

u, s, vt = svds(train_matrix, k = 20)

s_diag_matrix = np.diag(s)
predictions_svd = np.dot(np.dot(u,s_diag_matrix),vt)

predicted_ratings_svd = predictions_svd[test_matrix.nonzero()]
test_truth = test_matrix[test_matrix.nonzero()]
math.sqrt(mean_squared_error(predicted_ratings_svd,test_truth))

def gen_recom_svd(user_id, train_matrix):
    user_ratings = predictions_svd[user_id-1,:]
    train_unkown_indices = np.where(train_matrix[user_id-1,:] == 0)[0]
    user_recommendations = user_ratings[train_unkown_indices]

    print('\nRecommendations for user {} using SVD are the movies: '.format(user_id))
    for movie_id in user_recommendations.argsort()[-5:][: : -1]:
        print(np.array(movies.loc[movies['movie_id'] == movie_id+1].title))
    return

gen_recom_svd(user_id=886, train_matrix=train_matrix)
