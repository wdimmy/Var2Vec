# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import time
import argparse
from typing import Dict
from pprint import pprint

import torch
from torch import optim

from datasets import Dataset
from models import CP, ComplEx, DistMult
from regularizers import N2, N3
from optimizers import KBCOptimizer

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
	"""
	aggregate metrics for missing lhs and rhs
	:param mrrs: d
	:param hits:
	:return:
	"""
	m = (mrrs['lhs'] + mrrs['rhs']) / 2.
	h = (hits['lhs'] + hits['rhs']) / 2.
	return {'MRR': m, 'hits@[1,3,10]': h}


def train_kbc(KBC_optimizer, dataset, args):
	try:
		examples = torch.from_numpy(dataset.get_train(args.is_2i).astype('int64'))
	except:
		examples=torch.tensor(dataset.get_train(args.is_2i),dtype=torch.int64)
	max_epochs = args.max_epochs
	model_save_schedule = args.model_save_schedule

	cur_loss = 0
	curve = {'train': [], 'valid': [], 'test': []}
	warmup_step=max_epochs//10
	bar=10000000
	timestamp = str(int(time.time()))
	for epoch in range(1, max_epochs + 1):

		cur_loss = KBC_optimizer.train_epoch(examples)
		if epoch > bar:
			for param_group in KBC_optimizer.optimizer.param_groups:
				param_group["lr"]/=5
			bar+=warmup_step
		if (epoch + 1) % args.valid == 0:
			if not hasattr(KBC_optimizer.model,'is_matrix') or KBC_optimizer.model.is_matrix==-1:
				valid, test, train = [
					avg_both(*dataset.eval(KBC_optimizer.model, split, -1 if split != 'train' else 50000))
					for split in ['valid', 'test', 'train']
				]
			elif KBC_optimizer.model.is_2i==False:
				valid, test, train = [
					dataset.eval_dist(KBC_optimizer.model, split, -1 if split != 'train' else 50000)
					for split in ['valid', 'test', 'train']
				]
			else:
				valid, test, train = [
					dataset.eval_dist(KBC_optimizer.model, split, 100000)
					for split in ['valid_2i', 'test_2i', 'train_2i']
				]
			curve['valid'].append(valid)
			curve['test'].append(test)
			curve['train'].append(train)
			res={
				"train_loss":train,
				"valid_loss":valid,
				"time":int(time.time())
			}
			print("\t TRAIN: ", train)
			print("\t VALID : ", valid)

		if epoch % model_save_schedule == 0 and epoch > 0:
			if not os.path.isdir('models'):
				os.mkdir('models')
			if args.prefix:
				prefix=args.prefix
			else:
				prefix=models
			os.makedirs(args.prefix,exist_ok=True)
			model_dir = args.prefix
			torch.save({'epoch': epoch,
						'model_name': args.dataset,
						'factorizer_name': args.model,
						'regularizer': KBC_optimizer.regularizer,
						'optim_method': KBC_optimizer.optimizer,
						'batch_size': KBC_optimizer.batch_size,
						'model_state_dict': KBC_optimizer.model.state_dict(),
						'learning_rate':args.learning_rate,
						'optimizer_state_dict': KBC_optimizer.optimizer.state_dict(),
						'loss': cur_loss},
					    os.path.join(model_dir, '{}-{}-model-rank-{}-epoch-{}-lr-{}-layer-{}-{}.pt'.format(args.dataset, args.model, args.rank, epoch, args.learning_rate,args.is_matrix,timestamp)))

			with open(os.path.join(model_dir,'{}-metadata-{}.json'.format(args.dataset, timestamp)), 'w') as json_file:
				json.dump(vars(args), json_file)

	results = dataset.eval(model, 'test', -1)
	print("\n\nTEST : ", avg_both(*results))

	return curve, results


def kbc_model_load(model_path):
	"""
	This function loads the KBC model given the model. It uses the
	common identifiers in the name to identify the metadata/model files
	and load from there.

	@params:
		model_path - full or relative path to the model_path
	@returns:
		model : Class(KBCOptimizer)
		epoch : The epoch trained until (int)
		loss  : The last loss stored in the model
	"""
	identifiers = model_path.split('/')[-1]
	identifiers = identifiers.split('-')

	dataset_name, timestamp = identifiers[0].strip(), identifiers[-1][:-3].strip()
	if "YAGO" in dataset_name:
		dataset_name = "YAGO3-10"
	if 'FB15k' and '237' in identifiers:
		dataset_name = 'FB15k-237'

	model_dir = os.path.dirname(model_path)

	with open(os.path.join(model_dir, f'{dataset_name}-metadata-{timestamp}.json'), 'r') as json_file:
		metadata = json.load(json_file)

	map_location = None
	if not torch.cuda.is_available():
		print("yes")
		map_location = torch.device('cpu')
	

	checkpoint = torch.load(model_path, map_location=map_location)

	factorizer_name  = checkpoint['factorizer_name']
	models = ['CP', 'ComplEx', 'DistMult','transE']
	if 'cp' in factorizer_name.lower():
		model = CP(metadata['data_shape'], metadata['rank'], metadata['init'])
	elif 'complex' in factorizer_name.lower():
		if metadata.__contains__('is_matrix'):
			if type(metadata['is_matrix']) is bool:
				if metadata['is_matrix'] == True:
					is_matrix = 0
				else:
					is_matrix = -1
			else:
				is_matrix = metadata['is_matrix']
		else:
			is_matrix = -1
		model = ComplEx(metadata['data_shape'], metadata['rank'], metadata['init'], is_matrix,(metadata['is_2i'] if metadata.__contains__('is_2i') else None))
#		model = ComplEx(metadata['data_shape'], metadata['rank'], metadata['init'], metadata['is_matrix'])
	elif 'distmult' in factorizer_name.lower():
		if metadata.__contains__('is_matrix'):
			if type(metadata['is_matrix']) is bool:
				if metadata['is_matrix'] == True:
					is_matrix = 0
				else:
					is_matrix = -1
			else:
				is_matrix = metadata['is_matrix']
		else:
			is_matrix = -1
		model = DistMult(metadata['data_shape'], metadata['rank'], metadata['init'],is_matrix)
	elif 'transe' in factorizer_name.lower():
		model = transE(metadata['data_shape'], metadata['rank'], metadata['init'])
	else:
		raise ValueError(f'Model {factorizer_name} not in {models}')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	regularizer = checkpoint['regularizer']
	optim_method = checkpoint['optim_method']
	batch_size = checkpoint['batch_size']

	KBC_optimizer = KBCOptimizer(model, regularizer, optim_method, batch_size)
	KBC_optimizer.model.load_state_dict(checkpoint['model_state_dict'])
	KBC_optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	print(KBC_optimizer.model.eval())

	return KBC_optimizer, epoch, loss


def dataset_to_query(model, dataset_name, dataset_mode):
	queries = None
	try:
		dataset = Dataset(dataset_name)


		query_ids = dataset.dataset_to_queries(dataset_mode) # dataset_mode = [train,test,valid]

		query_side = []

		for i in range(len(query_ids[:,0])):
			query_side.append((query_ids[:,0][i].item(),query_ids[:,1][i].item()))

		check = []

		for i,j in query_side:
			check.append(dataset.to_skip['rhs'][i,j])

		queries = model.get_queries_separated(query_ids)

		if not('train' in dataset_mode.lower()):
			results =  dataset.eval(model, dataset_mode, -1)
			print("\n\n{} : {}".format(dataset_mode, results))


	except RuntimeError as e:
		print("Cannot convert the dataset to a query list with error: {}".format(str(e)))
		return None, None

	return queries,check

if __name__ == "__main__":

	modes = ['train', 'load']
	big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', "Bio"]
	datasets = big_datasets



	parser = argparse.ArgumentParser(
		description="Relational learning contraption"
	)


	parser.add_argument(
		'path'
	)

	models = ['CP', 'ComplEx', 'DistMult','transE']
	parser.add_argument(
		'--model', choices=models, default='ComplEx',
		help="Model in {}".format(models)
	)

	regularizers = ['N3', 'N2']
	parser.add_argument(
		'--regularizer', choices=regularizers, default='N3',
		help="Regularizer in {}".format(regularizers)
	)

	optimizers = ['Adagrad', 'Adam', 'SGD']
	parser.add_argument(
		'--optimizer', choices=optimizers, default='Adagrad',
		help="Optimizer in {}".format(optimizers)
	)

	parser.add_argument(
		'--max_epochs', default=50, type=int,
		help="Number of epochs."
	)
	parser.add_argument(
		'--valid', default=3, type=float,
		help="Number of epochs before valid."
	)
	parser.add_argument(
		'--rank', default=1000, type=int,
		help="Factorization rank."
	)
	parser.add_argument(
		'--batch_size', default=1000, type=int,
		help="Batch size."
	)
	parser.add_argument(
		'--reg', default=0, type=float,
		help="Regularization weight"
	)
	parser.add_argument(
		'--init', default=1e-3, type=float,
		help="Initial scale"
	)
	parser.add_argument(
		'--learning_rate', default=1e-1, type=float,
		help="Learning rate"
	)
	parser.add_argument(
		'--decay1', default=0.9, type=float,
		help="decay rate for the first moment estimate in Adam"
	)
	parser.add_argument(
		'--decay2', default=0.999, type=float,
		help="decay rate for second moment estimate in Adam"
	)

	parser.add_argument(
		'--model_save_schedule', default=50, type=int,
		help="Saving the model every N iterations"
	)
	parser.add_argument(
		'--is_matrix', default=-1, type=int,
		help="whethter to use matrix embedding"
	)
	parser.add_argument(
		'--is_2i', default=False, type=bool,
		help="whethter to use matrix embedding"
	)
	parser.add_argument(
		'--pretrained_model', default=None, type=str,
		help="load model"
	)
	parser.add_argument(
		'--prefix', default="models", type=str,
		help="load model"
	)
	parser.add_argument(
		'--device', default="cpu", type=str,
		help="load model"
	)
	parser.add_argument('--eval_only', action='store_true', default=False)
	parser.add_argument('--checkpoint', type=str)

	args = parser.parse_args()

	args.dataset = os.path.basename(args.path)

	dataset = Dataset(os.path.join(args.path, 'kbc_data'))
	args.data_shape = dataset.get_shape()

	if not args.eval_only:
		model = {
			'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
			'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init,args.is_matrix,args.is_2i),
			'DistMult': lambda: DistMult(dataset.get_shape(), args.rank, args.init,args.is_matrix),
			'transE': lambda: transE(dataset.get_shape(), args.rank, args.init)
		}[args.model]()
		if args.pretrained_model is not None:
			model.load_state_dict(torch.load(args.pretrained_model,map_location=torch.device('cpu'))['model_state_dict'],strict=False)
		regularizer = {
			'N2': N2(args.reg),
			'N3': N3(args.reg),
		}[args.regularizer]
		if not torch.cuda.is_available():
			device = 'cpu'
		else:
			device='cuda'
		model.to(device)

		if args.is_matrix != -1:
			for name,param in model.named_parameters():
				if "converter" not in name:
					param.requires_grad=False
		if args.is_2i is True:
			for name,param in model.named_parameters():
				if "tconverter" not in name:
					param.requires_grad=False

		print(vars(args))
#		wandb.init(project="cqd",entity="chenyeyuan",name=args.path+'cp'+str(args.is_matrix))
		optim_method = {
			'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
			'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
			'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
		}[args.optimizer]()

		KBC_optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)

		curve, results = train_kbc(KBC_optimizer,dataset,args)
	else:
		kbc, epoch, loss = kbc_model_load(args.checkpoint,args.is_matrix)
		for split in ['valid', 'test']:
			results = dataset.eval(kbc.model, split, -1)
			print(f"{split}: ", avg_both(*results))
