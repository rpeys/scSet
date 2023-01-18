#! /bin/bash

num_pcs=20
max_outputs=700
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4
adata_layer="pca"
batch_size=8
pid_col="pid"

lr=1e-3
beta=1e-2
epochs=2000
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=sadefeldman
log_name=gen/sadefeldman/
sadefeldman_data="/data/rna_rep_learning/sadefeldman/processed_adata_sparse.h5ad"

deepspeed train.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${num_pcs} \
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
  --log_name ${log_name} \
  --data_name ${data_name} \
  --h5ad_loc ${sadefeldman_data} \
  --num_pcs ${num_pcs} \
  --pid_col ${pid_col}
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
  --deepspeed_config batch_size_8.json \
  --val_recon_only

echo "Done"
exit 0
