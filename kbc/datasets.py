# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import pickle
import random
from typing import Dict, Tuple, List

import torch
from models import KBCModel
import numpy

class Dataset(object):
    def __init__(self, path):
        self.root = Path(path)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.txt.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)
        for f in ['train','test','valid']:
            in_file=open(path[:-9]+'/'+f+'_ans_2i.pkl','rb')
            data=pickle.load(in_file)
            self.data[f+'_2i']=[]
            for key,val in data.items():
                assert(len(key[0][1])==1)
                assert(len(key[1][1])==1)
                for ans in val:
                    self.data[f+'_2i'].append([key[0][0],key[0][1][0],key[1][0],key[1][1][0],ans])
            random.shuffle(self.data[f+'_2i'])
#            self.data[f+'_2i']=numpy.ndarray(self.data[f+'_2i'],dtype=numpy.int64)

        with open(str(self.root / 'ent_id.pickle'), 'rb') as f:
            self.n_entities = len(pickle.load(f))
        with open(str(self.root / 'rel_id.pickle'), 'rb') as f:
            self.n_predicates = len(pickle.load(f))

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self,is_2i=False):
        if is_2i is False:
            return self.data['train']
        else:
            return self.data['train_2i']
    def eval_dist(self, model: KBCModel, split: str, n_queries: int = -1):
 #       torch.cuda.empty_cache()
        test = self.get_examples(split)
        try:
            examples = torch.from_numpy(test.astype('int64'))
        except:
            examples=torch.tensor(test,dtype=torch.int64)
        q = examples.clone()
        if n_queries > 0:
            permutation = torch.randperm(len(examples))[:n_queries]
            q = examples[permutation]
        with torch.no_grad():
            loss,factors=model(q)
        return {"loss":torch.mean(loss)}
        

        return mean_reciprocal_rank, hits_at
    def eval(self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',at: Tuple[int] = (1, 3, 10)):

        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64'))
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp

                # Note: in q2b relations are labeled as
                # [rel1, rel1inv, rel2, rel2inv, ...]
                # In contrast, KBC uses
                # [rel1, rel2, ..., rel1inv, rel2inv, ...]
                # That's the reason for this:
                rels = q[:, 1].clone()
                q[:, 1][rels % 2 == 0] += 1
                q[:, 1][rels % 2 != 0] -= 1
                # Instead of:
                # q[:, 1] += self.n_predicates // 2

            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(lambda x: torch.mean((ranks <= x).float()).item(),at))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

    def dataset_to_queries(self,split: str):
        try:
            test = self.get_examples(split)
            examples = torch.from_numpy(test.astype('int64'))
            missing = ['rhs']

            for m in missing:
                q = examples.clone()
                if 'train' in split.lower():
                    permutation = torch.randperm(len(examples))[:5000]
                    q = examples[permutation]
                # if m == 'lhs':
                #     tmp = torch.clone(q[:, 0])
                #     q[:, 0] = q[:, 2]
                #     q[:, 2] = tmp
                #     q[:, 1] += self.n_predicates // 2

        except Exception as e:
            print("Unable to segment queries from dataset with error {}".format(str(e)))
            return None

        return q
