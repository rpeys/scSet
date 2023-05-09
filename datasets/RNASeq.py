"""
Adjusted from script for Set-MNIST dataset
"""
import os
import numpy as np
import scanpy as sc

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class scData(torch.utils.data.Dataset):
    def __init__(self, adata=None, name="unnamed_rnaseqdata", pid_col="pid", cat_col=None, pids=None, num_components=None, celltypes=None, cache_dir=None, adata_layer=None):
        """
        :param name: name of the dataset
        :param pids: subset of patient ids to use (used for creating train/val sets)
        """
        self.adata = adata
        self.name = name
        self.num_components = num_components #the number of principal components to use
        self.pids = pids
        self.pid_col=pid_col
        if self.pids is None:
           self.pids = adata.obs[pid_col].unique() #use all PIDs if none are specified
        self.cat_col=cat_col
        self.adata_layer = adata_layer
        self.celltypes = celltypes
        self.in_tr_sample_size = None
        self.in_te_sample_size = None
        self.maxcells = self.adata.obs[pid_col].value_counts().max() #max cells across all patients, not only the PIDs passed in
        self.subdirs = None
        self.scale = None
        self.random_subsample = None
        self.cache_dir = cache_dir
        if cache_dir is None:
            self.cache_dir = os.getcwd() 
            print("setting cache_dir: " + self.cache_dir)
        self.datasets = self._process_cache()

    def _process_cache(self):
        cache_path = os.path.join(self.cache_dir, "{}.pth".format(self.name))
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        os.makedirs(self.cache_dir, exist_ok=True)

        print("Processing dataset {}...".format(self.name))
        data = []
        for idx, pid in enumerate(self.pids): #train/test inds subset the PIDs
            pid_adata = self.adata[self.adata.obs[self.pid_col]==pid,:]
            # s is a tensor of size [N, Di] where N is the max number of cells and Di is the number of principal components
            if self.adata_layer == "pca":
                s = torch.zeros(self.maxcells, self.num_components)
                s[:len(pid_adata),:] = torch.from_numpy(pid_adata.obsm['X_pca'][:,:self.num_components])
            elif self.adata_layer == "hvg_lognorm":
                s = torch.zeros(self.maxcells, pid_adata.var.highly_variable.sum())
                s[:len(pid_adata),:] = torch.from_numpy(pid_adata.layers['lognorm'][:,pid_adata.var.highly_variable].todense())
            elif self.adata_layer == "hvg_raw":
                raise NotImplementedError
            else:
                raise ValueError("Must supply one of 'pca', 'hvg_lognorm', or 'hvg_raw' for adata_layer but {} was supplied.".format(self.adata_layer))
            # s_mask:[N, Di] True where there is no cell, False where there is a cell
            s_mask = torch.ones(self.maxcells, dtype=torch.bool)
            s_mask[:len(pid_adata)] = False
            #going to use mid to keep track of patient category. not sure what it was originally intended for...
            if self.cat_col is not None:
                mid = pid_adata.obs[self.cat_col][0]
            else:
                mid = None
            data.append({
                'idx': idx,
                'pid': pid,
                'set': s, 'mask': s_mask,
                'mean': 0, 'std': 1,      #'response': pid_adata.obs.response[0], 'therapy': pid_adata.obs.therapy[0]      
                'sid':None, 'mid':mid
                }) 
        torch.save(data, cache_path)
        print("Done! Saved data to %s" % cache_path)
        return data

    @staticmethod
    def get_pc_stats(idx):
        return 0., 1.

    def renormalize(mean, std):
        pass

    def save_statistics(self, save_dir):
        pass

    def __len__(self):
        return len(self.datasets) #if self.split == 'train' else len(self.test_points)

    def __getitem__(self, idx):
        return self.datasets[idx] #if self.split == 'train' else self.test_points[idx]


def collate_fn(batch):
    ret = dict()
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    s = torch.stack(ret['set'], dim=0)  # [B, N, input_dim]
    mask = torch.stack(ret['mask'], dim=0).bool()  # [B, N]
    cardinality = (~mask).long().sum(dim=-1)  # [B,]

    ret.update({'set': s, 'set_mask': mask, 'cardinality': cardinality,
                'mean': 0., 'std': 1.})
                
    return ret


def build(args):
    full_adata = sc.read_h5ad(args.h5ad_loc)
    pid_col=args.pid_col
    num_pids = len(full_adata.obs[pid_col].unique())
    print("num_pids: " + str(num_pids))
    train_pids, val_pids = torch.utils.data.random_split(full_adata.obs[pid_col].unique(), [round(0.8*num_pids), round(0.2*num_pids)], generator=torch.Generator().manual_seed(0))
    train_dataset = scData(adata=full_adata, name=args.data_name+"_train", pid_col=pid_col, pids=train_pids, cat_col=args.cat_col, num_components=args.input_dim, cache_dir=os.path.dirname(args.cache_dir), adata_layer=args.adata_layer)
    val_dataset = scData(adata=full_adata, name=args.data_name+"_val", pid_col=pid_col, pids=val_pids, cat_col=args.cat_col, num_components=args.input_dim, cache_dir=os.path.dirname(args.cache_dir), adata_layer=args.adata_layer)


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, worker_init_fn=init_np_seed)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                            pin_memory=True, drop_last=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, worker_init_fn=init_np_seed)

    return train_dataset, val_dataset, train_loader, val_loader
