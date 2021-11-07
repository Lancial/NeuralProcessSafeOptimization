import numpy as np
import pandas as pd

def load_user_movie_rating():
    data = pd.read_csv('ml-100k/u.data', sep="\t", header=None)
    data.columns = ['user id', 'movie id', 'rating', 'timestamp']
    return data


def load__movies_info():
    item = pd.read_csv('ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    item.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
                    'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    return item
    
def load_users_info():
    user = pd.read_csv('ml-100k/u.user', sep="|", encoding='latin-1', header=None)
    user.columns = ['user id', 'age', 'gender', 'occupation', 'zip code']
    return user
