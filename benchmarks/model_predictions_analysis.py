import os
import re
import glob
import fire
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

fields = 'a0_sym,a0_stct,a0_score,token_sim,deep_f_sim,deep_tag_sim,f_len_q,f_len_d'
fields = fields.split(',')

def mut_info_analysis(csv_file, n_rows=40_000):
    df = pd.read_csv(csv_file)
    print('Data rows:', len(df))
    print('Sample rows:', n_rows)
    df = df[:n_rows]

    global fields
    X = df[fields].to_numpy()
    y = df['y_pred'].to_numpy()

    mut_info = mutual_info_regression(X, y)
    return mut_info


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
