import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def reduce_movie_features(movie_features, k=20):
    movie_features = movie_features.drop(['IMDb URL', 'video release date', 'release date', 'movie title', 'movie id'], axis=1)
    X = movie_features.to_numpy()
    pca = PCA(n_components=k)
    return pca, pca.fit_transform(X)


def reduce_user_features(user_features, k=20):
    user_features["gender"] = user_features["gender"].astype('category')
    user_features["gender"] = user_features["gender"].cat.codes
    user_features["occupation"] = user_features["occupation"].astype('category')
    user_features["occupation"] = user_features["occupation"].cat.codes
    user_features = user_features.drop(["zip code", 'user id'], axis=1)
    X = user_features.to_numpy().astype(float)
    X = np.append(X, np.zeros((X.shape[0], k - X.shape[1])), axis=1)
    pca = PCA(n_components=k)
    return pca, pca.fit_transform(X)

def utility_matrix(user_movie_ratings):
    index=list(user_movie_ratings['user id'].unique())
    columns=list(user_movie_ratings['movie id'].unique())
    index=sorted(index)
    columns=sorted(columns)

    util_df=pd.pivot_table(data=user_movie_ratings,values='rating',index='user id',columns='movie id')
    return util_df