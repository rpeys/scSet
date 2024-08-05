#! /bin/bash

input_dim=50
max_outputs=700 #if you know the max n cells in your dataset, and it's << 2500 (the default), set this to that number
hidden_dim=64 
num_heads=4
adata_layer="pca"
batch_size=32
pid_col="patient"
cat_col="group"
num_seeds=100
#save_freq=10

lr=1e-4
epochs=100
scheduler="linear"
dataset_type=rnaseq
data_name=syn_props_tsubtypes_0.05vs0.15_dirstr50_10k
model_name=syn_props_tsubtypes_0.05vs0.15_dirstr50_10k_settransformer_v1
h5ad_loc="/data/rna_rep_learning/scset/synthetic_props_data/props_cd8_vs_cd4_0.05vs0.15_dirconc50_10000patients_HVGonly.h5ad"
cache_dir="/data/rna_rep_learning/scset/synthetic_props_data/"
num_workers=2 #run out of CPU memory with 4

deepspeed --include=localhost:0,1 --master_port 8080 train_settransformer.py \
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
  #--init_dim ${init_dim} \
  #--n_mixtures ${n_mixtures} \
  #--z_dim ${z_dim} \
  #--z_scales 4 8 16 32\
  #--save_freq ${save_freq} \
  #--viz_freq 10 \
  #--log_freq 10 \
  #--val_freq 10 \
  #--seed 42 \
  #  --beta ${beta} \
#  --val_recon_only \
#  --no_zigzag
