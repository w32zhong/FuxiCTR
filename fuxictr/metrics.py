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


def calc_auc(df, col_name):
    scores = df[col_name].to_numpy()
    clicks = df['click'].to_numpy()
    AUC = metrics.roc_auc_score(clicks, scores)
    return AUC


def calc_mrr(df, col_name):
    def apply_func(x_df):
        x_df = x_df.sort_values(by=[col_name], ascending=False)
        x_df = x_df.reset_index(drop=True)
        if x_df['click'].sum() == 0:
            return -1
        else:
            return x_df['click'].ne(0).idxmax() + 1
    mrr_df_group = df.groupby(['qid'])[['click', col_name]]
    mrr_df = mrr_df_group.apply(apply_func)
    ranks = mrr_df.to_numpy()
    ranks = ranks[ranks != -1]
    reciprocal_ranks = 1 / ranks
    meanRR = reciprocal_ranks.mean()
    return meanRR


def calc_map(df, col_name):
    #df = df[:10] ### DEBUG
    def apply_func(x_df):
        x_df = x_df.sort_values(by=[col_name], ascending=False)
        x_df = x_df.reset_index(drop=True)
        x_df['clicks_above'] = x_df['click'].cumsum()
        x_df = x_df.eval('precision = clicks_above / (index + 1)')
        if x_df['click'].sum() == 0:
            MAP = -1
        else:
            MAP = x_df['precision'].sum() / len(x_df)
        return MAP
    df_group = df.groupby(['qid'])[['click', col_name]]
    df = df_group.apply(apply_func)
    AP = df.to_numpy()
    AP = AP[AP != -1]
    MAP = AP.mean()
    return MAP


def calc_ndcg(df, col_name):
    #df = df[:10] ### DEBUG
    def apply_func(x_df):
        x_df = x_df.sort_values(by=[col_name], ascending=False)
        scores = x_df[col_name].to_numpy().reshape(1, -1)
        clicks = x_df['click'].to_numpy().reshape(1, -1)
        if clicks.shape[-1] == 1:
            NDCG = clicks[0][0]
        else:
            NDCG = metrics.ndcg_score(clicks, scores)
        return NDCG
    df_group = df.groupby(['qid'])[['rank', 'click', col_name]]
    df = df_group.apply(apply_func)
    NDCG = df.mean()
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
                result['NDCG'] = calc_ndcg(df, 'y_pred')

            elif metric == "MRR":
                result[metric] = calc_mrr(df, 'y_pred')

            elif metric == "HitRate":
                pass
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result
