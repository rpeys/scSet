#! /bin/bash

input_dim=50
max_outputs=700
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=32
num_heads=4
adata_layer="pca"
batch_size=32
pid_col="patient"
cat_col="group"
save_freq=10

lr=1e-4
beta=1 #1e-2
epochs=200
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=syn_props_tsubtypes_0.05vs0.15_dirstr50_10k
model_name=syn_props_tsubtypes_0.05vs0.15_dirstr50_10k_nozigzag_lr1e-4_supervised_4layers
h5ad_loc="/data/rna_rep_learning/scset/synthetic_props_data/props_cd8_vs_cd4_0.05vs0.15_dirconc50_10000patients_HVGonly.h5ad"
cache_dir="/data/rna_rep_learning/scset/synthetic_props_data/"
num_workers=4

deepspeed --include=localhost:2,3 --master_port 8080 train_supervised.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${input_dim} \
  --batch_size ${batch_size} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 4 8 16 32\
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
  --num_workers ${num_workers} \
  --no_zigzag
echo "Done"
exit 0
