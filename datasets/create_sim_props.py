import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scvi
import pickle
import os
import scipy
import seaborn as sns
from datetime import datetime
import torch
from utils import sample_pt_cells_scvi, viz_props
print("cuda available: " + str(torch.cuda.is_available()))

#eventual args
image_dir = "/home/rpeyser/GitHub/scSet/datasets/figures/simulated_props/"
data_name = "props_exp_scvi_tsubtypes_0.1vs0.05_dirstr30_1000pts"

image_dir = image_dir + data_name + "/"
if not os.path.exists(image_dir):
   # Create a new directory because it does not exist
   os.makedirs(image_dir)

sc.settings.figdir = image_dir

cd45_adata = sc.read_h5ad("/localdata/rna_rep_learning/zavidij_etal/cd45_adata.h5ad")

#temporarily convert to more specific cell types
cd45_adata.obs.celltype = cd45_adata.obs.Tcellsubtype.astype('str').replace('nan',np.NaN).fillna(cd45_adata.obs.celltype.astype('str'))

#setup data for scvi
scvi.model.SCVI.setup_anndata(cd45_adata)

#load trained scvi model
with open("/localdata/rna_rep_learning/scset/scvi_data/cd45model.pkl", "rb") as f:
    fullmodel = pickle.load(f)   
    
#generate synthetic data
cell_types = ["CD4 Cytotoxic", "CD8 Cytotoxic", "Helper 1", "Memory Cytotoxic", "B-cells", "CD14+ Monocytes"]
dir_strength = 30
cell_type_dirichlet_concentrations_1={"CD4 Cytotoxic":0.1*dir_strength, "CD8 Cytotoxic":0.05*dir_strength, "Helper 1":0.2*dir_strength, "Memory Cytotoxic":0.2*dir_strength, "B-cells":0.2*dir_strength, "CD14+ Monocytes":0.25*dir_strength}
cell_type_dirichlet_concentrations_2={"CD4 Cytotoxic":0.05*dir_strength, "CD8 Cytotoxic":0.1*dir_strength, "Helper 1":0.2*dir_strength, "Memory Cytotoxic":0.2*dir_strength, "B-cells":0.2*dir_strength, "CD14+ Monocytes":0.25*dir_strength}
npatients = 1000
mean_ncells = 445
total_cells_pp = scipy.stats.poisson.rvs(mean_ncells, size=npatients)
sim_counts = np.empty((np.sum(total_cells_pp), cd45_adata.shape[1]))
verbose=False

print("generating {} synthetic samples...".format(npatients))
cell_index = 0
for i in np.arange(npatients):
    if i % 100 == 0:
        print(i)
    if i <=(npatients/2-1):    
        sim_counts[cell_index:cell_index+total_cells_pp[i],:], joint_obs_Tsubtypes = sample_pt_cells_scvi(cd45_adata, fullmodel, cell_types, cell_type_dirichlet_concentrations_1, total_cells_pp[i], ptname="sim_pt{}".format(i+1), groupname="group1")

    else:
        sim_counts[cell_index:cell_index+total_cells_pp[i],:], joint_obs_Tsubtypes = sample_pt_cells_scvi(cd45_adata, fullmodel, cell_types, cell_type_dirichlet_concentrations_2, total_cells_pp[i], ptname="sim_pt{}".format(i+1), groupname="group2")
        
    cell_index += total_cells_pp[i]
    if i==0:
        sim_metadata = joint_obs_Tsubtypes.copy()
    else: 
        sim_metadata = pd.concat([sim_metadata, joint_obs_Tsubtypes], axis=0)

print("creating clustermaps of data...")
# clustering patients based on cell type counts
celltype_counts = sim_metadata[['celltype','group','patient']].groupby(["group","patient","celltype"]).size().unstack()
sns.clustermap(celltype_counts.reset_index().drop("group", axis=1).set_index("patient").fillna(0), row_colors=celltype_counts.reset_index().set_index("patient").group.map({"group1":"purple", "group2":"yellow"}), yticklabels=False, figsize=(7,7))
plt.savefig(image_dir + "clustermap_celltypecounts.png")      

print("creating stacked barplots of cell type proportions...")
viz_props(sim_metadata)
plt.savefig(image_dir + "stackedbar_celltypefrac.png", bbox_inches='tight')

# pseudobulk of these sample
pseudobulk_counts = pd.DataFrame(sim_counts, index=sim_metadata.patient, columns=cd45_adata.var.index).reset_index().groupby("patient").sum()
pseudobulk_lognorm = np.log1p(np.divide(pseudobulk_counts.T, pseudobulk_counts.sum(axis=1)))

# top 100 variable genes
gene_stds = pseudobulk_lognorm.std(axis=1)
topvargenes = gene_stds.sort_values()[-100:].index

sns.clustermap(pseudobulk_lognorm.loc[topvargenes,:], figsize=(7,7), col_colors=sim_metadata[['patient','group']].drop_duplicates().set_index('patient').group.map({'group1':'purple', 'group2':'yellow'}))
plt.savefig(image_dir + "clustermap_pseudobulk.png")        

print("creating anndata object...")
# make anndata object for model to use
sim_adata = sc.AnnData(sim_counts, obs=sim_metadata, var = cd45_adata.var)
sim_adata.layers['counts'] = sim_adata.X.copy()

print("lognorm data...")
sc.pp.normalize_total(sim_adata, target_sum=1e4)
sc.pp.log1p(sim_adata)
sc.pp.highly_variable_genes(sim_adata, min_mean=0.0125, max_mean=6, min_disp=0.5)
sim_adata.layers['lognorm'] = sim_adata.X.copy()

print("scale data and calculate PCA...")
sc.pp.scale(sim_adata, max_value=10)
sc.tl.pca(sim_adata, svd_solver='arpack')

#plot UMAP
print("calculate and plot UMAP...")
sc.pp.neighbors(sim_adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(sim_adata)
sc.pl.umap(sim_adata, color=['celltype', 'group'], ncols=1, save="_" + data_name + ".png")

# save adata
print("saving anndata object at the following location:")
print("/localdata/rna_rep_learning/scset/{}.h5ad".format(data_name))
sim_adata.write_h5ad("/localdata/rna_rep_learning/scset/{}.h5ad".format(data_name))

print("done!")


