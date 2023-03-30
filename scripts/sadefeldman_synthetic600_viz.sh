num_pcs=20
max_outputs=700
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4
adata_layer="pca"
batch_size=8 
pid_col="syn_pid"

lr=1e-3
beta=1 #1e-2
epochs=2000
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=syn_sadefeldman_noised_std0.73
log_name=/data/rna_rep_learning/scset/checkpoints/syn_sadefeldman_beta1_noised_std0.73_zscales_8_16_32_fullgenepca/
sadefeldman_data="/data/rna_rep_learning/sadefeldman/synthetic_data/syn_adata_noised_std_0.7261.h5ad"
cache_dir="/data/rna_rep_learning/sadefeldman/synthetic_data_noised_std0.73/"

python sample_and_summarize.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${num_pcs} \
  --batch_size ${batch_size} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --lr ${lr} \
  --beta ${beta} \
  --dataset_type ${dataset_type} \
  --log_name ${log_name} \
  --data_name ${data_name} \
  --h5ad_loc ${sadefeldman_data} \
  --cache_dir ${cache_dir} \
  --num_pcs ${num_pcs} \
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
  --deepspeed_config batch_size_8.json \
  --val_recon_only

echo "Done"
exit 0
