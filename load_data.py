import numpy as np
import pandas as pd

def load_user_movie_rating():
    data = pd.read_csv('ml-100k/u.data', sep="\t", header=None)
    data.columns = ['user id', 'movie id', 'rating', 'timestamp']
    return data


def load__movies_info():
    movie_features = pd.read_csv('ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    movie_features.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
                    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
#     movies = movie_features['movie id'].unique()
#     movieid2idx = {o:i for i,o in enumerate(movies)}
#     movie_features['movie id'] = movie_features['movie id'].apply(lambda x: movieid2idx[x])
    return movie_features
    
def load_users_info():
    user_features = pd.read_csv('ml-100k/u.user', sep="|", encoding='latin-1', header=None)
    user_features.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
#     users = user_features['user id'].unique()
#     userid2idx = {o:i for i,o in enumerate(users)}
#     user_features['user id'] = user_features['user id'].apply(lambda x: userid2idx[x])
    return user_features
