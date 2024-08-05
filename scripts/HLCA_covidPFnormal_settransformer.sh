#! /bin/bash

input_dim=50
max_outputs=700
hidden_dim=64
num_heads=4
adata_layer="pca"
batch_size=32
pid_col="sample"
cat_col="disease"
num_seeds=100

lr=1e-4
epochs=100
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=hlca_covidPFnormal_PCAinput
model_name=hlca_covidPFnormal_PCAinput_settransformer
h5ad_loc="/data/rna_rep_learning/hlca_sikkema2023/hlca_HVGonly_w_scGPT_embeds_SAMPLEPOOL_for_covidPFnormal_n213.h5ad"
cache_dir="/data/rna_rep_learning/scset/hlca_sikkema2023/"
num_workers=4

deepspeed --include=localhost:0,1,2,3 --master_port 8080 train_settransformer.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${input_dim} \
  --batch_size ${batch_size} \
  --max_outputs ${max_outputs} \
  --num_heads ${num_heads} \
  --num_seeds ${num_seeds} \
  --hidden_dim ${hidden_dim} \
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
