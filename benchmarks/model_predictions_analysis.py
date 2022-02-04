import os
import re
import glob
import fire
import pickle
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.special import digamma

import warnings
warnings.filterwarnings('error')

fields = 'a0_sym,a0_stct,a0_score,token_sim,deep_f_sim,deep_tag_sim,f_len_q,f_len_d'
fields = fields.split(',')

field_priority = 'deep_f_sim,deep_tag_sim,token_sim,a0_sym,a0_score,a0_stct,f_len_q,f_len_d'
field_priority = field_priority.split(',')

# Reference: https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py

# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    return x + intens * np.random.random_sample(x.shape)
def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]
def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)
def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))
def build_tree(points):
    return KDTree(points, metric='chebyshev')

# IM functions
def entropy(x, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    try:
        const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
        H = (const + n_features * np.log(nn).mean()) / np.log(base)
    except Warning:
        return -999
    return H


def cond_entropy(X, Y):
    XY = np.hstack((X, Y))
    return entropy(XY) - entropy(Y)


def mutual_info(X, Y):
    #return entropy(X) - cond_entropy(X, Y)
    return entropy(Y) - cond_entropy(Y, X)


def cond_mutual_info(X, Y, Z):
    if len(Z) == 0:
        return mutual_info(X, Y)
    YZ = np.hstack((Y, Z))
    return mutual_info(X, YZ) - mutual_info(X, Z)


def mut_info_analysis(csv_file, n_rows=40_000, top_k_feat=3):
    df = pd.read_csv(csv_file)
    print('Data rows:', len(df))
    print('Sample rows:', n_rows)
    df = df[:n_rows]
    global fields

    Y = df['y_pred'].to_numpy().reshape(-1, 1)
    Z_set = []
    feat_set = fields.copy()
    results = list()
    for k in range(top_k_feat):
        field_cmi = []
        for field in feat_set:
            X = df[field].to_numpy().reshape(-1, 1)
            Z = np.asarray(Z_set).T
            cmi = cond_mutual_info(Y, X, Z)
            field_cmi.append(cmi)
        max_field_idx = np.asarray(field_cmi).argmax()
        max_field = feat_set[max_field_idx]
        del feat_set[max_field_idx]
        max_cmi = max(field_cmi)
        print(f'top-{k} max field:', max_field, '=', max_cmi)
        results.append((max_field, max_cmi))
        Z_set.append(df[max_field].to_numpy())
    return results


def get_all_mut_info(root_dir):
    all_mut_info = dict()
    for path in glob.glob(f'{root_dir}/model_predictions-*.csv'):
        m = re.search(r"model_predictions-(\w+)_base.csv", path)
        model_name = m.group(1)
        mut_info = mut_info_analysis(path)
        all_mut_info[model_name] = mut_info
        print(model_name, mut_info)
    global fields
    results = (fields, all_mut_info)
    print(results)
    print('Saving results ...')
    with open('all_mut_info.pkl', 'wb') as fh:
        pickle.dump(results, fh)


def plot_subgraph(ax, mi_dict, filter_set):
    selected_models = list()
    x_labels = set()
    for model in mi_dict:
        important_feats = mi_dict[model]
        Y = [cmi for feat, cmi in important_feats]
        if model not in filter_set:
            continue
        elif max(Y) > 500:
            continue
        selected_models.append(model)
        feats = [feat for feat, cmi in important_feats]
        x_labels = x_labels.union(set(feats))
    x_labels = list(x_labels)
    x_labels = sorted(x_labels, key=lambda x: field_priority.index(x))

    for model in selected_models:
        important_feats = mi_dict[model]
        X = [x_labels.index(feat) for feat, cmi in important_feats]
        Y = [cmi for feat, cmi in important_feats]
        Y = list(map(lambda x: 0 if x < 0 else x, Y))
        ax.plot(X, Y, label=model, linestyle='solid')
        ax.scatter(X[0], Y[0], marker='*')
        ax.scatter(X[1], Y[1], marker='o')
        ax.scatter(X[2], Y[2], marker='v')

    x_ticks_labels = [
        t.replace('deep_f_sim', 'deep_sim') for t in x_labels
    ]
    return x_ticks_labels


def visualize_all_mut_info(pkl_file='all_mut_info.pkl'):
    with open(pkl_file, 'rb') as fh:
        all_mut_info = pickle.load(fh)

    fields, mi_dict = all_mut_info

    under_performs = 'FGCNN,FFMv2,AFM,AFN,InterHAt,CCPM,FM,FFM'.split(',')
    over_performs = 'DCNv2,FwFM,WideDeep,FiGNN,PNN,DeepFM,AutoInt,DCN,NFM,xDeepFM,FNN,DeepCrossing,ONN,DNN,DeepIM'.split(',')

    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=0.25)

    x_ticks_labels_0 = plot_subgraph(axs[0], mi_dict, over_performs)
    x_ticks_labels_1 = plot_subgraph(axs[1], mi_dict, under_performs)

    axs[0].legend(loc="upper right", framealpha=0)
    axs[0].set_xticks([i for i, _ in enumerate(x_ticks_labels_0)])
    axs[0].set_xticklabels(x_ticks_labels_0, rotation=0)

    axs[1].legend(loc='upper right', framealpha=0)
    axs[1].set_xticks([i for i, _ in enumerate(x_ticks_labels_1)])
    axs[1].set_xticklabels(x_ticks_labels_1, rotation=0)

    plt.show()


if __name__ == "__main__":
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "mut_info_analysis": mut_info_analysis,
        "get_all_mut_info": get_all_mut_info,
        "visualize_all_mut_info": visualize_all_mut_info
    })
