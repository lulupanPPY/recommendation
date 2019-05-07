from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from urllib import request

proxy_handler = request.ProxyHandler({'http': '100.129.58.13:3128'})
opener = request.build_opener(proxy_handler)
request.install_opener(opener)


# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)