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
import numpy
import logging


def calc_mmr(df, col_name):
    def apply_func(x_df):
        x_df = x_df.sort_values(by=[col_name], ascending=False)
        top_rank = x_df['rank'].iloc[0]
        return top_rank
    mmr_df_group = df.groupby(['qid', 'click'])[['rank', col_name]]
    mmr_df = mmr_df_group.apply(apply_func)
    ranks = mmr_df.loc[:, 1]
    reciprocal_ranks = 1 / ranks
    meanRR = reciprocal_ranks.mean()
    return meanRR


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
            if 'do_mmr' not in kwargs or kwargs['do_mmr'] is None:
                continue
            tsv_path = os.path.join(kwargs['do_mmr'], 'test.csv')
            df = pd.read_csv(tsv_path, skipinitialspace=True)
            df = df.join(pd.DataFrame(y_pred, columns=['y_pred']))
            if metric == "GAUC":
                pass
            elif metric == "NDCG":
                pass
            elif metric == "MRR":
                #with numpy.printoptions(threshold=numpy.inf):
                #import pdb; pdb.set_trace()

                file_y_true = df['click'].to_numpy()
                y_true = y_true.astype(int)
                assert (y_true == file_y_true).all()
                result[metric] = calc_mmr(df, 'y_pred')
            elif metric == "HitRate":
                pass
    logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
    return result
