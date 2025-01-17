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


import pandas as pd
import numpy as np
import os
from ..features import FeatureEncoder as BaseFeatureEncoder
from datetime import datetime, date

class FeatureEncoder(BaseFeatureEncoder):
    def qualify_shorthand_numbers(self, df, col_name):
        tens = dict(k=1e3, m=1e6, b=1e9)
        def _qualify_shorthand_numbers(x):
            x = x.strip()
            if len(x) == 0:
                return 0
            x = x.replace(',', '')
            if not x[-1].isdigit():
                y = int(int(x[:-1]) * tens[x[-1].lower()])
            else:
                y = int(x)
            return y
        nums = df[col_name].apply(_qualify_shorthand_numbers)
        return nums
