#! /bin/bash

input_dim=20
max_outputs=700
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4
adata_layer="pca"
batch_size=8 
pid_col="patient"

lr=1e-3
beta=1 #1e-2
epochs=1000
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=syn_pheno
model_name=syn_pheno
log_dir=gen/syn_pheno
h5ad_loc="/localdata/rna_rep_learning/scset/synthetic_pheno_exp/pheno_exp_adata.h5ad"
cache_dir="/localdata/rna_rep_learning/scset/synthetic_pheno_exp/"
num_workers=2 #run out of CPU memory with 4

python sample_and_summarize.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${input_dim} \
  --batch_size ${batch_size} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --lr ${lr} \
  --beta ${beta} \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --adata_layer ${adata_layer} \
  --cache_dir ${cache_dir} \
  --log_dir ${log_dir} \
  --data_name ${data_name} \
  --model_name ${model_name} \
  --h5ad_loc ${h5ad_loc} \
  --pid_col ${pid_col} \
  --resume_optimizer \
  --save_freq 100 \
  --viz_freq 10 \
  --log_freq 10 \
  --val_freq 10 \
  --scheduler ${scheduler} \
  --slot_att \
  --ln \
  --seed 42 \
  --distributed \
  --val_recon_only \
  --deepspeed_config batch_size.json \
  --num_workers ${num_workers}
echo "Done"
exit 0
