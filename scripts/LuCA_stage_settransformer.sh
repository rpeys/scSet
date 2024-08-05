#! /bin/bash

input_dim=10
max_outputs=700
hidden_dim=64
num_heads=4
adata_layer="X_scANVI"
batch_size=32
pid_col="sample"
cat_col="tumor_stage"
num_seeds=100
lr=1e-4
epochs=100
scheduler="linear"
dataset_type=rnaseq
data_name=luca_earlyvsadvanced
model_name=luca_earlyvsadvanced_settransformer_v1
h5ad_loc="/data/rna_rep_learning/luca_salcher2022/luca_salcher2022_SAMPLEPOOL_lungonly_earlyvsadvanced_n254.h5ad"
cache_dir="/data/rna_rep_learning/scset/luca_salcher2022/"
num_workers=4

deepspeed --include=localhost:2,3 --master_port 8000 train_settransformer.py \
  --input_dim ${input_dim} \
  --batch_size ${batch_size} \
  --max_outputs ${max_outputs} \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --num_seeds ${num_seeds} \
  --lr ${lr} \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --adata_layer ${adata_layer} \
  --cache_dir ${cache_dir} \
  --data_name ${data_name} \
  --model_name ${model_name} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --cat_col ${cat_col} \
  --resume_optimizer \
  --scheduler ${scheduler} \
  --slot_att \
  --ln \
  --distributed \
  --deepspeed_config batch_size.json \
  --num_workers ${num_workers}
echo "Done"
exit 0
