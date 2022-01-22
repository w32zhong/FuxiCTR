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


import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append('../')
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch import models
from fuxictr.pytorch.torch_utils import seed_everything
import gc
import argparse
import logging
import os
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='FM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--force', type=int, default=0, help='')
    parser.add_argument('--rerank', type=int, default=0, help='')
    parser.add_argument('--epochs', type=int, default=0, help='')
    
    args = vars(parser.parse_args())
    experiment_id = args['expid']

    touch_id = 'touch/' + experiment_id
    if os.path.exists(touch_id) and not args['force']:
        quit(0)

    params = load_config(args['config'], experiment_id, do_rerank=args['rerank'])
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info(print_to_json(params))
    seed_everything(seed=params['seed'])

    # preporcess the dataset
    dataset = params['dataset_id'].split('_')[0].lower()
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    if params.get("data_format") == 'h5': # load data from h5
        feature_map = FeatureMap(params['dataset_id'], data_dir, params['version'])
        json_file = os.path.join(os.path.join(params['data_root'], params['dataset_id']), "feature_map.json")
        if os.path.exists(json_file):
            feature_map.load(json_file)
        else:
            raise RuntimeError('feature_map not exist!')
    else: # load data from csv
        try:
            ds = getattr(datasets, dataset)
        except:
            raise RuntimeError('Dataset={} not exist!'.format(dataset))
        feature_encoder = ds.FeatureEncoder(**params)
        if params.get('use_hdf5') and os.path.exists(feature_encoder.pickle_file):
            feature_encoder = feature_encoder.load_pickle(feature_encoder.pickle_file)
        else: # Build feature_map and transform h5 data
            datasets.build_dataset(feature_encoder, **params)
        params["train_data"] = os.path.join(data_dir, 'train*.h5')
        params["valid_data"] = os.path.join(data_dir, 'valid*.h5')
        params["test_data"] = os.path.join(data_dir, 'test*.h5')
        feature_map = feature_encoder.feature_map

    # get train and validation data
    train_gen, valid_gen = datasets.h5_generator(feature_map, stage='train', **params)

    # initialize model
    model_class = getattr(models, params['model'])
    model = model_class(feature_map, **params)
    # print number of parameters used in model
    model.count_parameters()
    # fit the model
    if args['epochs'] > 0:
        params['epochs'] = args['epochs']
    model.fit_generator(train_gen, validation_data=valid_gen, **params)

    # load the best model checkpoint
    logging.info("Load best model: {}".format(model.checkpoint))
    model.load_weights(model.checkpoint)

    if isinstance(model, getattr(models, 'LR')):
        model.show_a0_weights()

    # get evaluation results on validation
    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate_generator(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    # get evaluation results on test
    logging.info('******** Test evaluation ********')
    params['shuffle'] = False
    test_gen = datasets.h5_generator(feature_map, stage='test', **params)
    test_result = model.evaluate_generator(test_gen, csv_dir=params['data_root'])
    
    # save the results to csv
    with open(Path(args['config']).stem + '.csv', 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
    
    os.makedirs('touch/', exist_ok=True)
    if not os.path.exists(touch_id):
        os.mknod(touch_id)
