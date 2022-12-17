# Var2Vec

The code is for our AAAI2023 paper: Efficient Embeddings of Logical Variables for Query Answering over Incomplete Knowledge Graphs (To appear) by Dingmin Wang, Yeyuan Chen and Bernardo Cuenca Grau.




### 1. Install the requirements

Our implementation is based on CQD ([Complex Query Answering with Neural Link Predictors](https://arxiv.org/abs/2011.03459)) repo [at this link](https://github.com/pminervini/KGReasoning/).

We recommend creating a new environment:

```bash
% conda create --name cqd python=3.8 && conda activate cqd
% pip install -r requirements.txt
```

### 2. Download the data

We use 3 knowledge graphs: FB15k, FB15k-237, and NELL.
From the root of the repository, download and extract the files to obtain the folder `data`, containing the sets of triples and queries for each graph.

```bash
% wget http://data.neuralnoise.com/cqd-data.tgz
% tar xvf cqd-data.tgz
```

### 3. Download the models

Then you need neural link prediction models -- one for each of the datasets.
Our pre-trained neural link prediction models are available here:

```bash
% wget http://data.neuralnoise.com/cqd-models.tgz
% tar xvf cqd-models.tgz
```

### 4. Train your model based on existing link predictors.

Then you can do the following to freeze link predictor and train the efficient matrix.

First, look at the tenth line of ```kbc/models.py```，change this address to you own working folder

Then you can run ```bash test.sh```  to train your models. These models will be put into ```models/converter```

### 5. Answering queries with our method

You can refer to ```test_final.py```  to answer queries with different types and models.

For example:

```
['FB15k-model-rank-1000-epoch-100-1602520745.pt','FB15k','True','1_1','0.0','0.0','0.1','0']
```

It encodes the location of your model (address that is relative to ```models/converter```), and the dataset you want to use. ```1_1``` means you want to answer ```1p``` queries. Other items are hyperparameters explained in our code.

To reproduce our reported result, please run ```test_final.py``` directly.
