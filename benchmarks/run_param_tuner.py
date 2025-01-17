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


import sys
sys.path.append('../')
from datetime import datetime
import gc
import argparse
from fuxictr import autotuner 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, help='The config file for param tuning.')
    parser.add_argument('--tag', type=str, default=None, help='Which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu', nargs='+', default=[-1], help='The list of gpu indexes, -1 for cpu.')
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    version = args['version']
    tag = args['tag']

    # generate parameter space combinations
    try:
        config_dir = autotuner.enumerate_params(args['config'])
    except Exception as e:
        print(e, file=sys.stderr)
        quit(0)
    autotuner.grid_search(version, config_dir, gpu_list, tag)

