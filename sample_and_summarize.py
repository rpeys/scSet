import os
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm
import itertools

from args import get_args
from datasets import get_datasets
from models.networks import SetVAE

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def get_train_loader(args):
    train_dataset, _, train_loader, _ = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        train_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_loader.collate_fn,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def get_test_loader(args):
    _, val_dataset, _, val_loader = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        val_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_loader.collate_fn,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def collate(result):
    # Concat summary
    for k, v in result.items():
        if 'set' in k or 'mask' in k or k in ['std', 'mean']:
            if type(v[0]) != torch.Tensor:
                result[k] = torch.tensor(v)
            else:
                result[k] = torch.cat(v, 0)
        elif 'att' in k:
            # OUTER LOOP, Z, Tensor
            inner = len(v[0])
            outer = len(v)
            lst = list()
            for i in range(inner):
                # 2, HEAD, BATCH, CARD, IND
                lst.append(torch.cat([v[j][i] for j in range(outer)], 2))
            result[k] = lst
        elif k in ['posteriors', 'priors']:
            # OUTER LOOP, Z, Tensor
            inner = len(v[0])
            outer = len(v)
            lst = list()
            for i in range(inner):
                lst.append(torch.cat([v[j][i] for j in range(outer)], 0))
            result[k] = lst
        else:
            result[k] = v
    return result


def recon(model, args, data):
    idx_b, gt, gt_mask = data['idx'], data['set'], data['set_mask']
    gt = gt.cuda()
    gt_mask = gt_mask.to(gt.device)

    output = model(gt, gt_mask)
    enc_att, dec_att = output['alphas']
    # Batch, Cardinality, n_mixtures
    enc_att = [torch.stack(a, 0).cpu() for a in enc_att]
    dec_att = [torch.stack(a, 0).cpu() for a in dec_att]

    posteriors = [z.cpu() for z, _, _ in output['posteriors']]

    # TODO: attention to cpu

    result = {
        'recon_set': output['set'].cpu(),
        'recon_mask': output['set_mask'].cpu(),
        'posteriors': posteriors,
        'dec_att': dec_att,
        'enc_att': enc_att,
        'enc_hiddens': [h.cpu() for h in output['enc_hiddens']],
        'init_set': output['init_set'].cpu(),
        'dec_hiddens': [h.cpu() for h in output['dec_hiddens']],
        'dec_latents': [z.cpu() for z in output['dec_latents']],
    }
    return result


def sample(model, args, data):
    gt_c = data['cardinality']
    gt_c = gt_c.cuda()

    output = model.sample(gt_c)
    priors = [z.cpu() for z, _, _ in output['priors']]
    smp_att = [torch.stack(a, 0).cpu() for a in output['alphas']]
    result = {
        'smp_set': output['set'].cpu(),
        'smp_mask': output['set_mask'].cpu(),
        'smp_att': smp_att,
        'priors': priors,
    }
    return result


def train_recon(model, args):
    loader = get_train_loader(args)
    save_dir = os.path.dirname(args.resume_checkpoint)

    summary = dict()
    for idx, data in enumerate(tqdm(loader)):
        gt_result = {
            'gt_set': data['set'],
            'gt_mask': data['set_mask'],
            'mean': data['mean'],
            'std': data['std'],
            'sid': data['sid'],
            'mid': data['mid'],
            'pid': data['pid'],
            'cardinality': data['cardinality'],
        }
        result = dict()

        # recon needs : set, mask, enc_att, dec_att, posterior, gt_set, gt_mask
        recon_result = recon(model, args, data)
        result.update(recon_result)
        result.update(gt_result)
        if len(summary.keys()) == 0:
            for k in result.keys():
                summary[k] = []
        for k, v in result.items():
            summary[k].append(v)

    summary = collate(summary)

    summary_name = Path(save_dir) / f"summary_train_recon.pth"
    torch.save(summary, summary_name)
    print(summary_name)


def sample_and_recon(model, args):
    all_sample, all_sample_mask = None, None
    all_ref, all_ref_mask = None, None

    loader = get_test_loader(args)
    save_dir = os.path.dirname(args.resume_checkpoint)

    summary = dict()
    for idx, data in enumerate(tqdm(loader)):
        gt_result = {
            'gt_set': data['set'],
            'gt_mask': data['set_mask'],
            'mean': data['mean'],
            'std': data['std'],
            'sid': data['sid'],
            'mid': data['mid'],
            'pid': data['pid'],
            'cardinality': data['cardinality'],
        }
        result = dict()
        # sample needs : Set, Mask, Smp_att, prior
        smp_result = sample(model, args, data)

        # recon needs : set, mask, enc_att, dec_att, posterior, gt_set, gt_mask
        recon_result = recon(model, args, data)
        result.update(smp_result)
        result.update(recon_result)
        result.update(gt_result)
        if len(summary.keys()) == 0:
            for k in result.keys():
                summary[k] = []
        for k, v in result.items():
            summary[k].append(v)

    summary = collate(summary)

    eval_latents(summary)

    summary_name = Path(save_dir) / f"summary.pth"
    torch.save(summary, summary_name)
    print(summary_name)

def eval_latents(summary, nclusts=2):
    patient_cats = list(itertools.chain.from_iterable(summary['mid']))
    if patient_cats[0] is not None: #if no patient groups are defined, can't eval latents
        for layer in np.arange(len(summary['enc_hiddens'][0])):
            print("layer {}".format(layer))
            val_hiddens = torch.concat([batch[layer] for batch in summary['enc_hiddens']]) #final hidden rep in encoder for each patient in the val set
            val_hiddens = val_hiddens.reshape(val_hiddens.shape[0], -1)
            sil = silhouette_score(val_hiddens, patient_cats)
            print("sillhouette score: {}".format(round(sil,2)))
            
            clusters = KMeans(n_clusters=nclusts).fit_predict(val_hiddens)
            ari = adjusted_rand_score(patient_cats, clusters)
            print("ARI: {}".format(round(ari,2)))
            print("\n")

def main(args):
    model = SetVAE(args)
    model = model.cuda()

    save_dir = Path(args.log_dir) / "checkpoints" / args.model_name
    # default to checkpoint-best
    if args.resume_checkpoint is None and Path(Path(save_dir) / 'checkpoint-best.pt').exists():
        args.resume_checkpoint = os.path.join(save_dir, f'checkpoint-best.pt')
    # else default to checkpoint-latest to be backwards compatible to before we saved the best checkpoint
    elif args.resume_checkpoint is None and Path(Path(save_dir) / 'checkpoint-latest.pt').exists():
        args.resume_checkpoint = os.path.join(save_dir, f'checkpoint-latest.pt')
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)

    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError:
        print("Load failed, trying compatibility matching")
        ckpt = checkpoint['model']
        updated_ckpt = OrderedDict()
        for k, v in ckpt.items():
            k = k.split('.')
            k[0] = f"{k[0]}.module"
            k = '.'.join(k)
            updated_ckpt.update({k: v})
        model.load_state_dict(updated_ckpt)
        print("Load success")

    model.eval()
    with torch.no_grad():
        sample_and_recon(model, args)
        train_recon(model, args)


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
