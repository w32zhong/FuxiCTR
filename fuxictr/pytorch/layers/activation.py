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

import torch
from torch import nn

class Dice(nn.Module):
    def __init__(self, input_dim, alpha=0., eps=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + (1 - p) * self.alpha * X
        return output
