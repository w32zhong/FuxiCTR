import os
import re
import glob
import fire
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.special import digamma

fields = 'a0_sym,a0_stct,a0_score,token_sim,deep_f_sim,deep_tag_sim,f_len_q,f_len_d'
fields = fields.split(',')

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
    const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
    return (const + n_features * np.log(nn).mean()) / np.log(base)


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
        Z_set.append(df[max_field].to_numpy())


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


def plot_models(pkl_file='all_mut_info.pkl'):
    with open(pkl_file, 'rb') as fh:
        all_mut_info = pickle.load(fh)

    fields, mi_dict = all_mut_info

    under_performs = 'FFM,FmFM,CCPM,FiBiNET,DeepFM,FM'.split(',')
    over_performs = 'FFMv2,DCNv2,FwFM,WideDeep,FiGNN,ONNv2,PNN,AutoInt,DCN,FNN,DeepCrossing,InterHAt,DNN,DeepIM'.split(',')

    fig, axs = plt.subplots(2)
    #fig.subplots_adjust(hspace=0.75)

    for model in mi_dict:
        mi = mi_dict[model]
        assert len(mi) == len(fields)
        if model in under_performs:
            axs[0].plot(mi, label=model)
        elif model in over_performs:
            axs[1].plot(mi, label=model)
        else:
            print('skip neutral model:', model)
            continue

    fields = ['NULL'] + fields
    axs[1].set_xticklabels(fields, rotation=70)
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "mut_info_analysis": mut_info_analysis,
        "get_all_mut_info": get_all_mut_info,
        "plot_models": plot_models
    })
