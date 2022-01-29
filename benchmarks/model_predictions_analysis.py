import os
import re
import glob
import fire
import pickle
import pandas as pd
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


if __name__ == "__main__":
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "mut_info_analysis": mut_info_analysis,
        "get_all_mut_info": get_all_mut_info
    })
