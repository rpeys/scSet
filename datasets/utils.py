import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sample_pt_cells_scvi(adata, scvi_model, cell_types, cell_type_dirichlet_conc, total_cells, ptname, groupname):
    
    # all cell types must be in metadata
    assert np.all(pd.Series(cell_types).isin(adata.obs.celltype)), "not all cell types are found in adata.obs"
    
    sample_cell_type_probs = scipy.stats.dirichlet.rvs([cell_type_dirichlet_conc[c] for c in cell_types]).squeeze() #ensures cell_type_concs is indexed in same order as cell_types 
    cell_type_counts = dict(zip(cell_types, scipy.stats.multinomial.rvs(total_cells, sample_cell_type_probs)))
    
    ## get rid of cell types with zero counts
    cell_type_counts = dict([(c, cell_type_counts[c]) for c in cell_type_counts.keys() if cell_type_counts[c]!=0]) 
    cell_types = cell_type_counts.keys()

    ## sample B, T, mono in certain proportions
    #find corresponding inds in adata
    celltype_inds = dict([(c, np.arange(len(adata))[adata.obs.celltype==c]) for c in cell_types])
    
    #sample
    chosen_inds = dict([(c, np.random.choice(celltype_inds[c], size = cell_type_counts[c])) for c in cell_types])

    #generate samples from posterior predictive dist
    samples = dict([(c, scvi_model.posterior_predictive_sample(indices=chosen_inds[c], n_samples=1)) for c in cell_types])

    joint_obs = pd.concat([adata.obs.iloc[chosen_inds[c],:] for c in cell_types])[["patient","celltype","Tcellsubtype"]].reset_index().rename(columns={"index":"orig_cellbarcode", "patient":"patient_orig"})
    joint_obs["group"] = pd.Series(groupname).repeat(total_cells).reset_index(drop=True)
    joint_obs["patient"] = pd.Series(ptname).repeat(total_cells).reset_index(drop=True)
    return np.concatenate(list(samples.values()), axis=0), joint_obs



def simulate_patient_counts(cell_types, cell_type_concs, total_cells, gene_params, loglibrary_mu, loglibrary_sigma, pt_id, grouplabel, groupval, avg_perturb_fc=4, perturb_prob=0.8, var_names=None, perturbed_gene_set=None, verbose=True):
    """
    cell_types: list of strings with labels for the cell types. Must be present in keys of gene_params and cell_type_concs. 
    cell_type_concs: dictionary of relative likelihood of different cell types, parameter of Dirichlet-Multinomial distribution for generating n for each cell type. Keys should match "cell_types."
    total_cells: the total cells for this patient (likely drawn from a Poisson outside of this function)
    gene_params: a dictionary of Gamma dist parameters per genes, for each cell type of interest. 
    """

    #everything will be pulled in the order of "cell_types" arg
    
    sample_cell_type_probs = scipy.stats.dirichlet.rvs([cell_type_concs[i] for i in cell_types]).squeeze() #ensures cell_type_concs is indexed in same order as cell_types 
    cell_type_counts = scipy.stats.multinomial.rvs(total_cells, sample_cell_type_probs)
    if verbose:
        print("cell type counts: {}".format(cell_type_counts))
    ngenes = len(gene_params[cell_types[0]])
    #simulate library sizes
    libsizes = np.exp(scipy.stats.norm.rvs(loc=loglibrary_mu, scale=loglibrary_sigma, size=total_cells))
    metadata = pd.DataFrame({"cell_type":np.repeat(cell_types, cell_type_counts), "target_libsize":libsizes, "patient":np.repeat(pt_id, total_cells), grouplabel:np.repeat(groupval, total_cells)})
    full_sample_matrix = np.empty((0, ngenes))
    for k in np.arange(len(cell_types)): #iterate over each cell type
        cell_type = cell_types[k]
        if cell_type not in gene_params.keys():
            raise("requested cell type {} not present in gene_params.keys()".format(cell_type))
        if verbose:    
            print("Generating {} for this sample...".format(cell_type))
        cell_by_gene_mat = np.zeros((cell_type_counts[k], ngenes))
        for n in np.arange(cell_type_counts[k]): #simulate one cell at a time            
            #samples all genes for one cell; genes that were 0 everywhere in training data are kept as 0
            cell_by_gene_mat[n,~gene_params[cell_type].iloc[:,0].isna()] = scipy.stats.gamma.rvs(a=gene_params[cell_type].loc[~gene_params[cell_type].iloc[:,0].isna()].a,
                                                                                                 loc=gene_params[cell_type].loc[~gene_params[cell_type].iloc[:,0].isna()].location,
                                                                                                 scale=gene_params[cell_type].loc[~gene_params[cell_type].iloc[:,0].isna()].scale)
            
        #concatenate this cell type expression to full matrix for sample
        full_sample_matrix = np.concatenate((full_sample_matrix, cell_by_gene_mat), axis=0)
    #set negative and very small numbers to 0
    tol = 1e-16
    full_sample_matrix[np.absolute(full_sample_matrix)<tol]=0
    full_sample_matrix[full_sample_matrix<0]=0
    
    # make DataFrame to add columns names
    full_sample_matrix = pd.DataFrame(full_sample_matrix, columns=var_names)
    #option for fold change in gene rates
    if perturbed_gene_set is not None:
        intersect_perturbed_gene_set = np.intersect1d(perturbed_gene_set, full_sample_matrix.columns)
        if len(intersect_perturbed_gene_set)<len(perturbed_gene_set):
            if verbose:
                print("The following genes were not found in your data (did you remember to pass in var_names?) : ")
                print(np.setdiff1d(perturbed_gene_set, full_sample_matrix.columns))
                print("Perturbing {} genes.".format(len(intersect_perturbed_gene_set)))
        if not len(intersect_perturbed_gene_set)==0:         
            fc_vec = np.exp(scipy.stats.norm.rvs(np.log(avg_perturb_fc), 0.15, size=len(intersect_perturbed_gene_set))) #generate fold changes
            
            # only perturb approx 80% of the cancer genes; 20% remain unaffected at random
            fc_vec[np.random.rand(len(fc_vec))>perturb_prob] = 1
            
            full_fc_vec = np.ones(full_sample_matrix.shape[1])
            full_fc_vec[full_sample_matrix.columns.isin(intersect_perturbed_gene_set)] = fc_vec

            full_sample_matrix = np.multiply(full_sample_matrix, full_fc_vec)
    #renormalize each cell
    full_sample_matrix = np.divide(full_sample_matrix.T, full_sample_matrix.sum(axis=1)).T
    #multiply gene rates by library size and sample counts from Poisson
    full_sample_matrix = np.multiply(full_sample_matrix.T, libsizes).T
    full_sample_counts = scipy.stats.poisson.rvs(full_sample_matrix)
    return full_sample_counts, metadata

def viz_props(obs_df):
    #visualize proportions in this dataset
    obs_df.celltype.cat.remove_unused_categories(inplace=True)
    celltypecounts = obs_df.groupby(["patient", "celltype"]).count().reset_index().iloc[:,:3]
    celltypecounts = celltypecounts.rename(columns = {celltypecounts.columns[2]:"ncells"})
    #celltypecounts['pt_totalcells'] = 
    pt_totalcells = celltypecounts.groupby("patient").sum().iloc[:,[0]].reset_index().rename(columns={'ncells':'pt_totalcells'})
    celltypecounts = celltypecounts.merge(pt_totalcells, on="patient")
    celltypecounts['fraction'] = celltypecounts.ncells / celltypecounts.pt_totalcells
    celltypecounts
    
    fig, ax = plt.subplots(1,1, figsize=[5,3])

    toplot = celltypecounts[["patient","celltype","fraction"]].pivot(index="patient", columns="celltype", values="fraction").fillna(0).reset_index()
    toplot.plot(kind="bar", stacked="true", x="patient", ax=ax, xticks=[])
    plt.legend(loc="upper right", bbox_to_anchor=[1.5,1]);