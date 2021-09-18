import numpy as np
import torch
import torch.nn.functional as F
import warnings

def unsqueeze(x, dim=-1, n=1):
    "Same as `torch.unsqueeze` but can add `n` dims"
    for _ in range(n): x = x.unsqueeze(dim)
    return x

def _bbs2sizes(crops, init_sz, use_square=True):
    bb = crops.flip(1)
    szs = (bb[1]-bb[0])
    if use_square: szs = szs.max(0)[0][None].repeat((2,1))
    overs = (szs+bb[0])>init_sz
    bb[0][overs] = init_sz-szs[overs]
    lows = (bb[0]/float(init_sz))
    return lows,szs/float(init_sz)

def crop_resize(x, crops, new_sz):
    # NB assumes square inputs. Not tested for non-square anythings!
    bs = x.shape[0]
    lows,szs = _bbs2sizes(crops, x.shape[-1])
    if not isinstance(new_sz,(list,tuple)):
        new_sz = (new_sz,new_sz)
    id_mat = torch.tensor([[1.,0,0],[0,1,0]])[None].repeat((bs,1,1)).to(x.device)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        sp = F.affine_grid(id_mat, (bs,1,*new_sz))+1.
        grid = sp*unsqueeze(szs.t(),1,n=2)+unsqueeze(lows.t()*2.,1,n=2)
        return F.grid_sample(x, grid-1, mode='nearest')

def _px_bounds(x, dim):
    c = x.sum(dim).nonzero().cpu()
    idxs,vals = torch.unique(c[:,0],return_counts=True)
    vs = torch.split_with_sizes(c[:,1],tuple(vals))
    d = {k.item():v for k,v in zip(idxs,vs)}
    default_u = torch.tensor([0,x.shape[-1]-1])
    b = [d.get(o,default_u) for o in range(x.shape[0])]
    b = [torch.tensor([o.min(),o.max()]) for o in b]
    return torch.stack(b)

def mask2bbox(mask):
    no_batch = mask.dim()==2
    if no_batch: mask = mask[None]
    bb1 = _px_bounds(mask,-1).t()
    bb2 = _px_bounds(mask,-2).t()
    res = torch.stack([bb1,bb2],dim=1).to(mask.device)
    return res[...,0] if no_batch else res

def squarePad(x):
    long = max(x.shape[-2:])
    short = min(x.shape[-2:])
    if long == x.shape[3]:
        d3 = long
        d2 = long-short
        cat_dim = 2
    else:
        d2 = long
        d3 = long-short
        cat_dim = 3
    pad = torch.zeros((x.shape[0], x.shape[1], d2, d3), dtype=x.dtype, device = x.device)
    padded = torch.cat((x, pad), dim=cat_dim)
    return padded

def bbox2Square(bbox, pad=0):
    bbox_center = (bbox[:, :, 1] + bbox[:, :, 0]) / 2.0
    bbox_radius = ((bbox[:, :, 1] - bbox[:, :, 0]).max(-1)[0]) / 2.0 + pad
    bbox_square = bbox_center.unsqueeze(-1).repeat(1, 1, 2)
    bbox_square[:, :, 0] -= bbox_radius.unsqueeze(1)
    bbox_square[:, :, 1] += bbox_radius.unsqueeze(1)
    bbox_square = bbox_square.ceil()
    return bbox_square
