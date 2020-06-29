# Data preprocessing, since the data downloaded is in .dat format
# Import packages
import os
import pandas as pd

# define relative path for the files
relative_path = "./data/"
user_data_file = 'users.dat'
movie_data_file = 'movies.dat'
rating_data_file = 'ratings.dat'


# Specify User's Age and Occupation Column
AGES = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
               4: "college/grad student", 5: "customer service", 6: "doctor/health care",
               7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
               12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
               17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

# Define csv files to be saved into
user_csv_file = 'users.csv'
movie_csv_file = 'movies.csv'
rating_csv_file = 'ratings.csv'


# Read the Ratings File
ratings = pd.read_csv(relative_path + rating_data_file,
                      sep='::',
                      engine='python',   # Parser engine to use
                      encoding='latin-1',  # Encoding to use while reading file
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])


# Set max_userid to the maximum user_id in the ratings and also remove duplicates of user ID if there's any
max_userid = max(ratings['user_id'].drop_duplicates())

# Set max_movieid to the maximum movie_id in the ratings and also remove duplicates of user ID if there's any
max_movieid = max(ratings['movie_id'].drop_duplicates())

# Process ratings dataframe for Keras Deep Learning model
# Add user_emb_id column whose values == user_id - 1
ratings['user_emb_id'] = ratings['user_id'] - 1
# Add movie_emb_id column whose values == movie_id - 1
ratings['movie_emb_id'] = ratings['movie_id'] - 1

print(len(ratings), 'ratings loaded')

save_path = "./parsed data/"
ratings.to_csv(save_path + rating_csv_file,
               sep='\t',
               header=True,
               encoding='latin-1',
               columns=['user_id', 'movie_id', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
print('Saved to', rating_csv_file)

# Read the Users File
users = pd.read_csv(relative_path + user_data_file,
                    sep='::',
                    engine='python',
                    encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])

users['age_desc'] = users['age'].apply(lambda x: AGES[x])
users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])
print(len(users), 'descriptions of', max_userid, 'users loaded.')

# Save into users.csv
users.to_csv(save_path + user_csv_file,
             sep='\t',
             header=True,
             encoding='latin-1',
             columns=['user_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])
print('Saved to', user_csv_file)

# Read the Movies File
movies = pd.read_csv(relative_path + movie_data_file,
                     sep='::',
                     engine='python',
                     encoding='latin-1',
                     names=['movie_id', 'title', 'genres'])
print(len(movies), 'descriptions of', max_movieid, 'movies loaded.')

# Save into movies.csv
movies.to_csv(save_path + movie_csv_file,
              sep='\t',
              header=True,
              columns=['movie_id', 'title', 'genres'])
print('Saved to', movie_csv_file)
