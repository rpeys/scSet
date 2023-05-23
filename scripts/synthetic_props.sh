#! /bin/bash

input_dim=20
max_outputs=700
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=32
num_heads=4
adata_layer="pca"
batch_size=40
pid_col="patient"
cat_col="group"
save_freq=10

lr=1e-3
beta=1 #1e-2
epochs=600
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=syn_props_tsubtypes_0.05vs0.15_dirstr100
model_name=syn_props_hdim32_scvi_tsubtypes_0.05vs0.15_dirstr100
h5ad_loc="/localdata/rna_rep_learning/scset/props_exp_scvi_tsubtypes_0.05vs0.15_dirstr100.h5ad"
cache_dir="/localdata/rna_rep_learning/scset/synthetic_props_scvi/"
num_workers=2 #run out of CPU memory with 4

deepspeed --include=localhost:1,2 --master_port 8081 train.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${input_dim} \
  --batch_size ${batch_size} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 2 4 8 16 32\
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --lr ${lr} \
  --beta ${beta} \
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
  --save_freq ${save_freq} \
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
