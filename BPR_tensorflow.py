import numpy
import tensorflow as tf
import os
import random
from collections import defaultdict

epochs = 4

def load_data(data_path):
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
    print("max_u_id:", max_u_id)
    print("max_i_id:", max_i_id)
    return max_u_id, max_i_id, user_ratings

data_path = os.path.join('C://Users//panlu//lightfm_data//movielens100k//ml-100k', 'u.data')
user_count, item_count, user_ratings = load_data(data_path)

def generate_test(user_ratings):
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
        # this sample is not choosing subset from user_ratings but choose a randomly one movie for each user_rating item
    return user_test




def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512):
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
    while i == user_ratings_test[u]:
        i = random.sample(user_ratings[u], 1)[0]
    j = random.randint(1, item_count)
    while j in user_ratings[u]:
        j = random.randint(1, item_count)
    t.append([u, i, j])
    return numpy.asarray(t)

user_ratings_test = generate_test(user_ratings)

def generate_test_batch(user_ratings, user_ratings_test, item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count+1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
    yield numpy.asarray(t)


def bpr_mf(user_count, item_count, hidden_dim):
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    with tf.device("/cpu:0"):
        user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                     initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                     initializer=tf.random_normal_initializer(0, 0.1))

        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)

    # MF predict: u_i > u_j
    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    #
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)


    return u, i, j, mf_auc, bprloss, train_op


with tf.Graph().as_default(), tf.Session() as session:
    u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, 20)
    session.run(tf.initialize_all_variables())
    for epoch in range(1, epochs):
        _batch_bprloss = 0
        for k in range(1, 5000):  # uniform samples from training set
            uij = generate_train_batch(user_ratings, user_ratings_test, item_count)

            _bprloss, _train_op = session.run([bprloss, train_op],
                                      feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
            _batch_bprloss += _bprloss

        print("epoch: ", epoch)
        print("bpr_loss: ", _batch_bprloss / k)
        print("_train_op")

        user_count = 0
        _auc_sum = 0.0

# each batch will return only one user's auc
    for t_uij in generate_test_batch(user_ratings, user_ratings_test, item_count):
        _auc, _test_bprloss = session.run([mf_auc, bprloss],
                                      feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]}
                                      )
        user_count += 1
        _auc_sum += _auc
    print("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count)
    print("")
    variable_names = [v.name for v in tf.trainable_variables()]
    values = session.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)