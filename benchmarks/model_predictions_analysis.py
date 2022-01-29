import os
import fire
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def mut_info_analysis(csv_file='model_predictions-LR_base.csv', n_rows=40_000):
    df = pd.read_csv(csv_file)
    print('Data rows:', len(df))
    print('Sample rows:', n_rows)
    df = df[:n_rows]

    fields = 'a0_sym,a0_stct,a0_score,token_sim,deep_f_sim,deep_tag_sim,f_len_q,f_len_d'
    fields = fields.split(',')
    X = df[fields].to_numpy()
    y = df['y_pred'].to_numpy()

    mut_info = mutual_info_regression(X, y)
    print(dict(zip(fields, mut_info)))


if __name__ == "__main__":
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        "mut_info_analysis": mut_info_analysis
    })
