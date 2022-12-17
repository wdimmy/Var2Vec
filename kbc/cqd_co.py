#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import os.path as osp
import json
from tokenize import String
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

from utils import QuerDAG
from utils import preload_env
from metrics import evaluation
import numpy as np

def score_queries(args):
    mode = args.mode

    dataset = osp.basename(args.path)
    #data_hard_path='data/FB15k-237/kbc_data/train.txt.pickle'
    #data_complete_path='data/FB15k-237/kbc_data/train.txt.pickle'
    data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')
    data_hard_not_path=('/data/cqd/'+args.dataset+'-betae/'+'test-hard-answers.pkl')
    data_easy_not_path=('/data/cqd/'+args.dataset+'-betae/'+'test-easy-answers.pkl')
    data_not_queries_path=('/data/cqd/'+args.dataset+'-betae/'+'test-queries.pkl')
    import sys
    sys.path.append('/home/chenyeyuan/cqd/cqd')
    data_hard = pickle.load(open(data_hard_path, 'rb'))
    data_complete = pickle.load(open(data_complete_path, 'rb'))
    data_not_hard=pickle.load(open(data_hard_not_path, 'rb'))
    data_not_easy=pickle.load(open(data_easy_not_path, 'rb'))
    data_not_queries=pickle.load(open(data_not_queries_path, 'rb'))
    for key,val in data_not_hard.items():
        data_not_easy[key]=data_not_easy[key]|val
    # Instantiate singleton KBC object
    preload_env(args.model_path, data_hard, args.chain_type, mode='hard',explain=args.explain,extra_datasets=(data_not_hard,data_not_queries),switch=args.switch)
    env = preload_env(args.model_path, data_complete, args.chain_type,
                      mode='complete',explain=args.explain,extra_datasets=(data_not_easy,data_not_queries),switch=args.switch)

    queries = env.keys_hard
    test_ans_hard = env.target_ids_hard
    test_ans = env.target_ids_complete
    chains = env.chains
    kbc = env.kbc

    if args.reg is not None:
        env.kbc.regularizer.weight = args.reg

    disjunctive = args.chain_type in (QuerDAG.TYPE2_2_disj.value,
                                      QuerDAG.TYPE4_3_disj.value)
    save_flag=False
    if args.chain_type == QuerDAG.TYPE1_1.value:
        # scores = kbc.model.link_prediction(chains)

        s_emb = chains[0][0]
        p_emb = chains[0][1]

        scores_lst = []
        #embedding=torch.tensor([],dtype=torch.float32).cuda()
        nb_queries = s_emb.shape[0]
        for i in tqdm(range(nb_queries)):
            batch_s_emb = s_emb[i, :].view(1, -1)
            batch_p_emb = p_emb[i, :].view(1, -1)
            batch_chains = [(batch_s_emb, batch_p_emb, None)]
            batch_scores = kbc.model.link_prediction(batch_chains)
            if save_flag is True:
                embed=kbc.model.converter(torch.cat((batch_s_emb.squeeze(),batch_p_emb.squeeze()),0)).reshape(1,-1)
                embedding=torch.cat((embedding,embed),dim=0)
            scores_lst += [batch_scores]
        if save_flag is True:
            torch.save(embedding,'/data/cqd2/type1_1.pt')
            torch.save(kbc.model.embeddings[0],'/data/cqd2/embedding.pt')
        scores = torch.cat(scores_lst, 0)
        min_loss=0
        min_step=0
        dists=None

    elif args.chain_type in (QuerDAG.TYPE1_2.value, QuerDAG.TYPE1_3.value):
        scores, min_loss, min_step,dists = kbc.model.optimize_chains(chains, kbc.regularizer,
                                           max_steps=args.max_steps,
                                           lr=args.lr,
                                           optimizer=args.optimizer,
                                           norm_type=args.t_norm,
                                           matrix=args.matrix,
                                           bias1=args.bias1,
                                           bias2=args.bias2,
                                           bias3=args.bias3,
                                           half=args.half,
                                           batchsize=args.batchsize)

    elif args.chain_type in (QuerDAG.TYPE2_2.value, QuerDAG.TYPE2_2_disj.value,
                             QuerDAG.TYPE2_3.value):
        scores, min_loss, min_step,dists = kbc.model.optimize_intersections(chains, kbc.regularizer,
                                                  max_steps=args.max_steps,
                                                  lr=args.lr,
                                                  optimizer=args.optimizer,
                                                  norm_type=args.t_norm,
                                                  disjunctive=disjunctive,
                                                  matrix=args.matrix)

    elif args.chain_type in (QuerDAG.TYPE3_3.value,QuerDAG.TYPE3_3_not.value,QuerDAG.TYPE3_3_notdown.value) and not args.tmp:
        scores, min_loss,min_step,dists = kbc.model.optimize_3_3(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        matrix=args.matrix,
                                        type=args.chain_type,
                                        bias1=args.bias1,
                                        bias2=args.bias2)
    elif args.chain_type in (QuerDAG.TYPE3_3.value,QuerDAG.TYPE3_3_not.value,QuerDAG.TYPE3_3_notdown.value) and args.tmp:
        scores, min_loss,min_step,dists = kbc.model.optimize_3_3_tmp(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        matrix=args.matrix,
                                        type=args.chain_type)

    elif args.chain_type == QuerDAG.TYPE4_3_disj.value:
        scores, min_loss,min_step,dists = kbc.model.optimize_4_3(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        disjunctive=disjunctive,
                                        matrix=args.matrix,
                                        bias1=args.bias1,
                                        bias2=args.bias2,
                                        batchsize=args.batchsize)
    elif args.chain_type in (QuerDAG.TYPE4_3.value, QuerDAG.TYPE4_3_not.value):
         scores, min_loss,min_step,dists = kbc.model.optimize_4_3_consj(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        disjunctive=disjunctive,
                                        matrix=args.matrix,
                                        is_neg=(args.chain_type == QuerDAG.TYPE4_3_not.value),
                                        bias1=args.bias1,
                                        bias2=args.bias2,
                                        bias3=args.bias3)    
    elif args.chain_type in (QuerDAG.TYPE2_2_not.value,QuerDAG.TYPE2_3_not.value):
        scores, min_loss,min_step,dists = kbc.model.optimize_intersections_not(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        disjunctive=disjunctive,
                                        matrix=args.matrix)
    else:
        raise ValueError(f'Uknown query type {args.chain_type}')
    if args.half==True:
        queries=queries[len(queries)//2:]
    return scores, queries, test_ans, test_ans_hard, min_loss,min_step,dists


def main(args):
    scores, queries, test_ans, test_ans_hard,min_loss,min_step,dists = score_queries(args)
    metrics = evaluation(scores, queries, test_ans, test_ans_hard)
    metrics.update({"min_loss":min_loss,"min_step":min_step})
    if args.chain_type=='1_2' and dists is not None:
        metrics.update({"dist_a":dists[0],"dist_b":dists[1],"dist":dists[2]})
    elif args.chain_type=='1_3' and dists is not None:
        metrics.update({"dist_a":dists[0],"dist_b":dists[1],"dist_c":dists[2],"dist":dists[3]})
    print(metrics)

    model_name = osp.splitext(osp.basename(args.model_path))[0]
    reg_str = f'{args.reg}' if args.reg is not None else 'None'
    import os
    os.makedirs(args.prefix,exist_ok=True)
    with open(args.prefix+f'/cont_n={model_name}_t={args.chain_type}_r={reg_str}_m={args.mode}_lr={args.lr}_opt={args.optimizer}_ms={args.max_steps}.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":

    datasets = ['FB15k', 'FB15k-237', 'NELL']
    modes = ['valid', 'test', 'train']
    chain_types = [t.value for t in QuerDAG]

    t_norms = ['min', 'prod']

    parser = argparse.ArgumentParser(description="Complex Query Decomposition - Continuous Optimisation")
    parser.add_argument('path', help='Path to directory containing queries')
    parser.add_argument('--model_path', help="The path to the KBC model. Can be both relative and full")
    parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets))
    parser.add_argument('--mode', choices=modes, default='test',
                        help="Dataset validation mode in {}".format(modes))

    parser.add_argument('--chain_type', choices=chain_types, default=QuerDAG.TYPE1_1.value,
                        help="Chain type experimenting for ".format(chain_types))

    parser.add_argument('--t_norm', choices=t_norms, default='prod', help="T-norms available are ".format(t_norms))
    parser.add_argument('--reg', type=float, help='Regularization coefficient', default=None)
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batchsize', type=int, default=-1, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adagrad', 'sgd'])
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--explain', default=False,
						action='store_true',
						help='Generate log file with explanations for 2p queries')
    parser.add_argument('--matrix',type=int, default=0,
						help='Generate log file with explanations for 2p queries')
    parser.add_argument('--switch',default=False,
						action='store_true',help='whether to use another set of datasets')
    parser.add_argument('--tmp',default=False,
						action='store_true',help='whether to use another set of datasets')
    parser.add_argument('--pca',default=False,
						action='store_true',help='whether to use another set of datasets')
    parser.add_argument('--prefix',
						help='Generate log file with explanations for 2p queries')
    parser.add_argument('--bias1', type=float, default=0.0, help='Learning rate')
    parser.add_argument('--bias2', type=float, default=0.0, help='Learning rate')
    parser.add_argument('--bias3', type=float, default=0.0, help='Learning rate')
    parser.add_argument('--half',default=False,
						action='store_true',help='whether to use another set of datasets')
    main(parser.parse_args())
