import pandas as pd
import matplotlib.pyplot as plt
import string
import re
from afinn import Afinn
from scipy import sparse
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#users=pd.read_csv('RAW_interactions.csv')
#recipes=pd.read_csv('RAW_recipes.csv')
users1=pd.read_csv('rem_us.csv')

def sem_analyse(df):
    for index, row in df.iterrows():
        if (row['rating'] == 0):
            afn = Afinn()
            score = afn.score(row["review"])
            if score >= 5:
                df["rating"].iloc[index] = 5
            elif score > 0:
                df["rating"].iloc[index] = 4
            elif score == 0:
                df["rating"].iloc[index] = 3
            elif score <= -5:
                df["rating"].iloc[index] = 1
            else:
                df["rating"].iloc[index] = 2

def remove_neg_users(df):
    non = df["user_id"].value_counts().loc[lambda x: x == 1].to_frame().reset_index()
    non = non.rename({'index': 'user_id', 'user_id': 'count'}, axis='columns')
    one_rev = non['user_id'].tolist()
    for i in one_rev:
        if int(df["rating"].loc[df["user_id"]==i])<=3:
            df.drop(df.loc[df['user_id']==i].index, inplace=True)
def remove_users(df):
    non = df["user_id"].value_counts().loc[lambda x: x == 1].to_frame().reset_index()
    non = non.rename({'index': 'user_id', 'user_id': 'count'}, axis='columns')
    one_rev = non['user_id'].tolist()
    print(len(one_rev))
    for i in one_rev:
        print(i)
        df.drop(df.loc[df['user_id']==i].index, inplace=True)

def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.

    Args:
        df: pandas dataframe containing 3 columns (userId, movieId, rating)

    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['user_id'].nunique()
    N = df['recipe_id'].nunique()

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(M))))
    recipe_mapper = dict(zip(np.unique(df["recipe_id"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["user_id"])))
    recipe_inv_mapper = dict(zip(list(range(N)), np.unique(df["recipe_id"])))

    user_index = [user_mapper[i] for i in df['user_id']]
    item_index = [recipe_mapper[i] for i in df['recipe_id']]

    X = csr_matrix((df["rating"], (item_index,user_index)), shape=(N,M),dtype='uint8')

    return X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper

# sem_analyse(users)
#remove_neg_users(users1)


# df=users1.loc[users1["rating"]>3]
# non = users1["user_id"].value_counts().loc[lambda x: x == 1].to_frame().reset_index()
# #non= df["recipe_id"].value_counts().loc[lambda x: x == 1].to_frame().reset_index()
# # print(users1.shape)
# print(df.shape)
# #print(len(non))
# # non = df["user_id"].value_counts().loc[lambda x: x == 1].to_frame().reset_index()
# # print(len(non))
# remove_users(df)
# print(df.shape)
# df.to_csv(index=False)
# compression_opts = dict(method='zip',
#                         archive_name='rem_us.csv')
# df.to_csv('rem_us.zip', index=False,
#           compression=compression_opts)
#print(users1["rating"].nunique)

X, user_mapper, recipe_mapper, user_inv_mapper, recipe_inv_mapper = create_X(users1)
non = users1["recipe_id"].value_counts().loc[lambda x: x == 1].to_frame().reset_index()
df = pd.DataFrame(data=csr_matrix.todense(X))
print(users1['recipe_id'].nunique())
#print(df)
print(len(non))
#df=df.astype(np.uint8)
print(len(non))
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(df.values)
distances, indices = knn.kneighbors(df.values, n_neighbors=3)