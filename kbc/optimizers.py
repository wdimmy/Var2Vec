# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tqdm
import torch
from torch import nn
from torch import optim

from models import KBCModel
from regularizers import Regularizer



class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def train_epoch(self, examples: torch.LongTensor,loss_type=None):

        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                if self.model.embeddings[0].weight.device.type !='cpu':
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ].cuda()
                else:
                    input_batch = actual_examples[
                        b_begin:b_begin + self.batch_size
                    ]
                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]
                if len(predictions.shape)==1:
                    l_fit=torch.mean(predictions)
                else:
                    l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                #get the Clipping here

                b_begin += self.batch_size

                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.5f}')
