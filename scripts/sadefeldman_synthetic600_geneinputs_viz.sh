input_dim=3937 #PCA:20
max_outputs=700
init_dim=5300 #there are 5261 hvg. #for PCA: 32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4
adata_layer="hvg_lognorm"
batch_size=10
pid_col="syn_pid"

lr=1e-3
beta=1 #1e-2
epochs=2000
kl_warmup_epochs=50
scheduler="linear"
dataset_type=rnaseq
data_name=syn_sadefeldman_noised_std0.73_geneinputs
log_dir=/data/rna_rep_learning/scset/
model_name=syn_sadefeldman_beta1_noised_std0.73_geneinputs
sadefeldman_data="/data/rna_rep_learning/sadefeldman/synthetic_data/syn_adata_noised_std_0.7261.h5ad"
cache_dir="/data/rna_rep_learning/sadefeldman/synthetic_data_noised_std0.73_geneinputs/"


python sample_and_summarize.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${input_dim} \
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
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --adata_layer ${adata_layer} \
  --cache_dir ${cache_dir} \
  --log_dir ${log_dir} \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --h5ad_loc ${sadefeldman_data} \
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
  --deepspeed_config batch_size.json

echo "Done"
exit 0
