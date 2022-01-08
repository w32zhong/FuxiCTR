# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import os
import pandas as pd
import numpy as np
import logging
from sklearn import metrics


def calc_mrr(df, col_name):
    def apply_func(x_df):
        x_df = x_df.sort_values(by=[col_name], ascending=False)
        top_rank = x_df['rank'].iloc[0]
        return top_rank
    mrr_df_group = df.groupby(['qid', 'click'])[['rank', col_name]]
    mrr_df = mrr_df_group.apply(apply_func)
    ranks = mrr_df.loc[:, 1]
    reciprocal_ranks = 1 / ranks
    meanRR = reciprocal_ranks.mean()
    return meanRR


def calc_map(df, col_name):
    #df = df[:10] ### DEBUG
    def apply_func(x_df):
        x_df = x_df.sort_values(by=[col_name], ascending=False)
        x_df['clicks_above'] = x_df['click'].cumsum()
        x_df['index'] = pd.Series(1 + np.arange(len(x_df))).values
        x_df = x_df.eval('precision = clicks_above / index')
        x_df = x_df[x_df['click'] != 0]
        if len(x_df) == 0:
            MAP = 0
        else:
            MAP = x_df['precision'].sum() / len(x_df)
        return MAP
    df_group = df.groupby(['qid'])[['rank', 'click', col_name]]
    df = df_group.apply(apply_func)
    MAP = df.mean()
    return MAP


def calc_ndcg(df, col_name, k=None):
    #df = df[:10] ### DEBUG
    scores = df[col_name].to_numpy().reshape(1, -1)
    clicks = df['click'].to_numpy().reshape(1, -1)
    NDCG = metrics.ndcg_score(clicks, scores, k=k)
    return NDCG


def evaluate_metrics(y_true, y_pred, metrics, **kwargs):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            y_pred = np.argmax(y_pred, axis=1)
            result[metric] = accuracy_score(y_true, y_pred)
        else:
            if 'csv_dir' not in kwargs or kwargs['csv_dir'] is None:
                continue
            tsv_path = os.path.join(kwargs['csv_dir'], 'test.csv')
            df = pd.read_csv(tsv_path, skipinitialspace=True)
            df = df.join(pd.DataFrame(y_pred, columns=['y_pred']))

            file_y_true = df['click'].to_numpy()
            y_true = y_true.astype(int)
            assert (y_true == file_y_true).all()

            if metric == "MAP":
                result[metric] = calc_map(df, 'y_pred')

            elif metric == "NDCG":
                result['NDCG@10'] = calc_ndcg(df, 'y_pred', 10)
                result['NDCG@50'] = calc_ndcg(df, 'y_pred', 50)
                result['NDCG@100'] = calc_ndcg(df, 'y_pred', 100)
                result['NDCG@all'] = calc_ndcg(df, 'y_pred')

            elif metric == "MRR":
                result[metric] = calc_mrr(df, 'y_pred')

            elif metric == "HitRate":
                pass
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result
