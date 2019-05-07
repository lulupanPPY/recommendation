import pandas as pd
import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import operator

anime = pd.read_csv('D://Kaggle//anime-recommendations-database//anime.csv')
rating = pd.read_csv('D://Kaggle//anime-recommendations-database//rating.csv')

rating.rating.replace({-1: np.nan}, regex=True, inplace = True)
#print (rating.head())

anime_tv = anime[anime['type']=='TV']
#print (anime_tv.head())

merged = rating.merge(anime_tv, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)

merged=merged[['user_id', 'name', 'user_rating']]
merged_sub= merged[merged.user_id <= 10000]
print (merged_sub.head())
#pivot
piv = merged_sub.pivot_table(index=['user_id'], columns=['name'], values='user_rating')
#normalize
piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
#fill null value
piv_norm.fillna(0, inplace=True)
#remove users rate nothing, or anime no user rated
piv_norm = piv_norm.T
piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]
#turn into sparse matrix format
piv_sparse = sp.sparse.csr_matrix(piv_norm.values)
#similarities
item_similarity = cosine_similarity(piv_sparse)
user_similarity = cosine_similarity(piv_sparse.T)
item_sim_df = pd.DataFrame(item_similarity, index = piv_norm.index, columns = piv_norm.index)
user_sim_df = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)

# This function will return the top 10 shows with the highest cosine similarity value

def top_animes(anime_name):
    count = 1
    print('Similar shows to {} include:\n'.format(anime_name))
    for item in item_sim_df.sort_values(by = anime_name, ascending = False).index[1:11]:
        print('No. {}: {}'.format(count, item))
        count +=1


# This function will return the top 5 users with the highest similarity value

def top_users(user):
    if user not in piv_norm.columns:
        return ('No data available on user {}'.format(user))

    print('Most Similar Users:\n')
    sim_values = user_sim_df.sort_values(by=user, ascending=False).loc[:, user].tolist()[1:11]
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    zipped = zip(sim_users, sim_values, )
    for user, sim in zipped:
        print('User #{0}, Similarity value: {1:.2f}'.format(user, sim))


# This function constructs a list of lists containing the highest rated shows per similar user
# and returns the name of the show along with the frequency it appears in the list

def similar_user_recs(user):
    if user not in piv_norm.columns:
        return ('No data available on user {}'.format(user))

    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    best = []
    most_common = {}

    for i in sim_users:
        max_score = piv_norm.loc[:, i].max()
        best.append(piv_norm[piv_norm.loc[:, i] == max_score].index.tolist())
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1
    sorted_list = sorted(most_common.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list[:5]

# This function calculates the weighted average of similar users
# to determine a potential rating for an input user and show

def predicted_rating(anime_name, user):
    sim_users = user_sim_df.sort_values(by=user, ascending=False).index[1:1000]
    user_values = user_sim_df.sort_values(by=user, ascending=False).loc[:,user].tolist()[1:1000]
    rating_list = []
    weight_list = []
    for j, i in enumerate(sim_users):
        rating = piv.loc[i, anime_name]
        similarity = user_values[j]
        if np.isnan(rating):
            continue
        elif not np.isnan(rating):
            rating_list.append(rating*similarity)
            weight_list.append(similarity)
    return sum(rating_list)/sum(weight_list)


top_animes('Cowboy Bebop')
