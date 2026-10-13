"""Teach ORTrack to use several templates (option 2): light fine-tuning on GOT-10k.

The released ORTrack was trained with one template. Here it sees an ANCHOR template (an
earlier frame, as the frame-1 template at test time) plus 0-3 MEMORY views of the same target
from 5-80 frames before the search frame, all as one token sequence (as Backend.forward_multi).
Memory views are cut at JITTERED boxes so the network learns to cope with the slightly wrong
boxes a tracker produces. Trained: the attention layers of all blocks and a learned SLOT code
per template position (anchor, memory 1..3), added to that template's tokens so the network
can tell the templates apart. Slot codes start at zero: before training the network is exactly
the official ORTrack. Everything else (patch embedding, MLPs, norms, box head) stays frozen.
Loss: ORTrack's own (focal on the score map 1, GIoU 2, L1 5).

The trained parameters are saved to <out>/finetuned.pth; backends.Backend(..., finetuned=path)
loads them. The authors' code is not modified.

    python finetune_multi.py --iters 20000 --out results/finetune/ortrack_mt_s1
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/mnt/c/Users/justi/ORTrack/ORTrack')

from frenv import load_videos  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------- data
class MultiTemplateData(torch.utils.data.Dataset):
    """Sample i of batch b has n_mem = (i // batch) % (max_mem + 1) memory views, so every
    batch has one template count."""

    def __init__(self, root, json_name, length, batch, max_mem=3, min_mem=0, min_lag=5, max_lag=80, anchor_gap=400,
                 jitter_c=0.08, jitter_s=0.08, search_center_jitter=3.0, search_scale_jitter=0.25):
        from lib.train.data.processing_utils import sample_target, transform_image_to_crop
        self.sample_target, self.to_crop = sample_target, transform_image_to_crop
        self.videos = [v for v in load_videos(root, json_name) if len(v['frames']) >= 20]
        self.length, self.batch, self.max_mem, self.min_mem = length, batch, max_mem, min_mem
        self.min_lag, self.max_lag, self.anchor_gap = min_lag, max_lag, anchor_gap
        self.jc, self.js = jitter_c, jitter_s
        self.scj, self.ssj = search_center_jitter, search_scale_jitter
        self._absent = {}

    def __len__(self):
        return self.length

    def _visible(self, v):
        if v not in self._absent:
            vid = self.videos[v]
            d = os.path.dirname(vid['frames'][0])
            lab = os.path.join(d, 'absence.label')
            ab = np.loadtxt(lab).astype(bool) if os.path.exists(lab) else np.zeros(len(vid['frames']), bool)
            gt = vid['gt']
            ab = ab[:len(gt)] if len(ab) >= len(gt) else np.r_[ab, np.zeros(len(gt) - len(ab), bool)]
            self._absent[v] = ~ab & (gt[:, 2] > 1) & (gt[:, 3] > 1)
        return self._absent[v]

    @staticmethod
    def _img(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def _crop(self, img, box, factor, size):
        return self.sample_target(img, list(map(float, box)), factor, output_sz=size)

    def __getitem__(self, i):
        rng = np.random
        n_mem = self.min_mem + (i // self.batch) % (self.max_mem - self.min_mem + 1)
        while True:
            v = rng.randint(len(self.videos))
            vis = np.where(self._visible(v))[0]
            if len(vis) >= 10:
                break
        vid = self.videos[v]
        f = int(rng.choice(vis[vis >= min(vis[-1], self.min_lag)]))
        earlier = vis[(vis < f) & (vis >= f - self.anchor_gap)]
        a = int(rng.choice(earlier)) if len(earlier) else int(vis[0])
        # memory frames: distinct lags, nearest visible frame at or before f - lag
        mem = []
        lags = sorted(rng.choice(np.arange(self.min_lag, self.max_lag + 1), n_mem, replace=False)) if n_mem else []
        for lag in lags:
            cand = vis[vis <= f - lag]
            mem.append(int(cand[-1]) if len(cand) else a)
        gt = vid['gt']
        z = [self._crop(self._img(vid['frames'][a]), gt[a], 2.0, 128)[0]]
        for m in mem:
            b = gt[m].copy()
            s = np.sqrt(b[2] * b[3])
            c = b[:2] + b[2:] / 2 + rng.normal(0, self.jc * s, 2)
            wh = b[2:] * np.exp(rng.normal(0, self.js, 2))
            z.append(self._crop(self._img(vid['frames'][m]), np.r_[c - wh / 2, wh], 2.0, 128)[0])
        # search region: OSTrack-style jitter around the ground truth
        b = gt[f]
        wh = b[2:] * np.exp(rng.randn(2) * self.ssj)
        max_off = np.sqrt(wh.prod()) * self.scj
        c = b[:2] + b[2:] / 2 + max_off * (rng.rand(2) - 0.5)
        ext = np.r_[c - wh / 2, wh]
        x, rf = self._crop(self._img(vid['frames'][f]), ext, 4.0, 256)[:2]
        box = self.to_crop(torch.tensor(b, dtype=torch.float32), torch.tensor(ext, dtype=torch.float32), rf,
                           torch.tensor([256., 256.]), normalize=True)
        return {'z': np.stack(z), 'x': x, 'box': box.clamp(0, 1).numpy().astype(np.float32)}


# ---------------------------------------------------------------------------- model
class MultiTemplateNet(nn.Module):
    def __init__(self, network, n_slots=4, train_part='attn'):
        super(MultiTemplateNet, self).__init__()
        self.net = network
        dim = network.backbone.pos_embed_z.shape[-1]
        self.slot = nn.Parameter(torch.zeros(n_slots, 1, dim))
        for p in self.net.parameters():
            p.requires_grad = False
        for name, p in self.net.named_parameters():
            if train_part == 'attn' and name.startswith('backbone.blocks.') and '.attn.' in name:
                p.requires_grad = True
            elif train_part == 'blocks' and name.startswith('backbone.blocks.'):
                p.requires_grad = True

    def forward(self, zs, x):
        bb = self.net.backbone
        xt = bb.patch_embed(x) + bb.pos_embed_x
        zt = [bb.patch_embed(z) + bb.pos_embed_z + self.slot[j] for j, z in enumerate(zs)]
        h = bb.pos_drop(torch.cat(zt + [xt], dim=1))
        for blk in bb.blocks:
            h = blk(h)
        return self.net.forward_head(bb.norm(h), None)

    def trainable_state(self):
        keep = {k for k, p in self.net.named_parameters() if p.requires_grad}
        return {'net': {k: v for k, v in self.net.state_dict().items() if k in keep}, 'slot': self.slot.detach()}


class MemoryMerger(nn.Module):
    """Option 1d: compress up to `max_views` memory views (each 64 tokens) into ONE template's
    worth of tokens. Summary tokens start as the per-position average of the views and are
    refined by attention over all memory tokens (+ a learned code per view position, newest
    first); the refinement's output projection starts at zero, so an untrained merger is the
    plain average."""

    def __init__(self, dim, max_views=3, layers=2, heads=4):
        super(MemoryMerger, self).__init__()
        self.view_code = nn.Parameter(torch.zeros(max_views, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
                                  for _ in range(layers)])
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, views):
        """views: list of (B, L, dim) token tensors, newest first. Returns (B, L, dim)."""
        m = torch.stack(views, 1)                                   # B, V, L, dim
        base = m.mean(1)
        mem = (m + self.view_code[:m.shape[1]].unsqueeze(0)).flatten(1, 2)
        q = base
        for attn, n1, n2, mlp in zip(self.attn, self.n1, self.n2, self.mlp):
            k = n1(mem)
            q = q + attn(n1(q), k, k, need_weights=False)[0]
            q = q + mlp(n2(q))
        return base + self.out(q - base)


class MergerNet(nn.Module):
    """Frozen ORTrack + MemoryMerger: sequence [anchor | merged memory | search]."""

    def __init__(self, network, max_views=3):
        super(MergerNet, self).__init__()
        self.net = network
        for p in self.net.parameters():
            p.requires_grad = False
        self.merger = MemoryMerger(network.backbone.pos_embed_z.shape[-1], max_views)

    def forward(self, zs, x):
        bb = self.net.backbone
        xt = bb.patch_embed(x) + bb.pos_embed_x
        zt = [bb.patch_embed(z) + bb.pos_embed_z for z in zs]
        seq = [zt[0]] + ([self.merger(zt[1:])] if len(zt) > 1 else []) + [xt]
        h = bb.pos_drop(torch.cat(seq, dim=1))
        for blk in bb.blocks:
            h = blk(h)
        return self.net.forward_head(bb.norm(h), None)

    def trainable_state(self):
        return {'type': 'merger', 'merger': self.merger.state_dict(),
                'max_views': self.merger.view_code.shape[0]}


def norm(t):
    """uint8 (B, H, W, 3) -> normalised (B, 3, H, W) on the GPU."""
    t = t.cuda().float().permute(0, 3, 1, 2) / 255.
    return (t - MEAN.cuda()) / STD.cuda()


# ---------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/finetune/ortrack_mt_s1')
    ap.add_argument('--config', default='deit_tiny_patch16_224')
    ap.add_argument('--root', default='/home/justin/data/GOT-10k')
    ap.add_argument('--json', default='GOT10k_train.json')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--slot_lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--train_part', default='attn', choices=['attn', 'blocks'])
    ap.add_argument('--mode', default='slots', choices=['slots', 'merger'],
                    help='slots: option 2 (train attention + slot codes, 4 template blocks); merger: option 1d '
                         '(ORTrack frozen, learned merger compresses the memory views into one block)')
    ap.add_argument('--merger_lr', type=float, default=3e-4)
    ap.add_argument('--max_mem', type=int, default=3)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--save_every', type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'log.txt'), 'a')

    def say(msg):
        line = '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()
    say('args ' + json.dumps(vars(args)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from backends import Backend
    from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
    from lib.utils.focal_loss import FocalLoss
    from lib.utils.heapmap_utils import generate_heatmap
    backend = Backend('ortrack', args.config)
    if args.mode == 'merger':
        model = MergerNet(backend.t.network, max_views=args.max_mem).cuda()
        params = list(model.merger.parameters())
        say('trainable: {} merger parameters (ORTrack frozen)'.format(sum(p.numel() for p in params)))
        opt = torch.optim.AdamW([{'params': params, 'lr': args.merger_lr}], weight_decay=args.wd)
    else:
        model = MultiTemplateNet(backend.t.network, n_slots=args.max_mem + 1, train_part=args.train_part).cuda()
        params = [p for p in model.net.parameters() if p.requires_grad] + [model.slot]
        say('trainable: {} network parameters + {} slot parameters'.format(
            sum(p.numel() for p in params) - model.slot.numel(), model.slot.numel()))
        opt = torch.optim.AdamW([{'params': params[:-1], 'lr': args.lr}, {'params': [model.slot], 'lr': args.slot_lr}],
                                weight_decay=args.wd)
    model.eval()                                    # no dropout / drop-path in the frozen parts
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    focal = FocalLoss()
    data = MultiTemplateData(args.root, args.json, args.iters * args.batch, args.batch, max_mem=args.max_mem,
                             min_mem=1 if args.mode == 'merger' else 0)
    say('GOT-10k: {} videos'.format(len(data.videos)))
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch, shuffle=False, num_workers=args.workers, drop_last=True,
        worker_init_fn=lambda w: np.random.seed(args.seed * 1000 + w + int(time.time()) % 100000),
        persistent_workers=True, prefetch_factor=4)
    tic, run = time.time(), {}
    for it, batch in enumerate(loader, 1):
        z = batch['z']                                  # (B, n, 128, 128, 3) uint8
        zs = [norm(z[:, j]) for j in range(z.shape[1])]
        x = norm(batch['x'])
        gt = batch['box'].cuda()                        # (B, 4) normalised x, y, w, h
        out = model(zs, x)
        pred = box_cxcywh_to_xyxy(out['pred_boxes'].view(-1, 4))
        gtx = box_xywh_to_xyxy(gt).clamp(0, 1)
        gl, iou, _ = giou_loss(pred, gtx)
        l1 = F.l1_loss(pred, gtx)
        hm = generate_heatmap([gt.cpu()], 256, 16)[-1].unsqueeze(1).cuda()
        fl = focal(out['score_map'], hm)
        loss = 2 * gl + 5 * l1 + fl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.1 if args.mode == 'slots' else 1.0)
        opt.step()
        sched.step()
        n = z.shape[1] - 1
        for k, val in (('loss', loss.item()), ('iou', iou.mean().item())):
            run.setdefault((k, n), []).append(val)
        if it % 200 == 0:
            msg = '  '.join('mem{} loss {:.3f} iou {:.3f}'.format(
                m, np.mean(run[('loss', m)]), np.mean(run[('iou', m)])) for m in range(args.max_mem + 1)
                if ('loss', m) in run)
            say('iter {}/{}  {}  ({:.1f} it/s)'.format(it, args.iters, msg, 200 / (time.time() - tic)))
            tic, run = time.time(), {}
        if it % args.save_every == 0 or it == args.iters:
            torch.save(dict(model.trainable_state(), args=vars(args), iter=it),
                       os.path.join(args.out, 'finetuned.pth'))
        if it >= args.iters:
            break
    say('done')


if __name__ == '__main__':
    main()
