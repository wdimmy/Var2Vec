model_name=(
    FB15k-237-model-rank-1000-epoch-100-1602508358.pt
    FB15k-model-rank-1000-epoch-100-1602520745.pt
    NELL-model-rank-1000-epoch-100-1602499096.pt
)
data_name=(
    FB15k-237
    FB15k
    NELL
)
rank=(
    1000
    1000
    1000
    1000
    1000
    1000
)
cuda_id=(0 0 0)
id=(0 1 2)
lr=(0.1 0.1 0.1)
for i in ${id[@]}
do
    echo ${data_name[i]}
    python kbc/learn.py data/${data_name[i]} --model ComplEx --pretrained_model models/${model_name[i]} --is_matrix 0 --rank ${rank[i]} --reg 0.01 --max_epochs 100 --batch_size 1000 --learning_rate 0.1 --prefix models/converter --is_matrix 0
done


