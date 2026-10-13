"""Batched access to the released UETrack / ORTrack trackers for the failure-recovery work.

Each tracker's own objects are used unchanged (network, preprocessing, Hanning window,
crop and box functions); only the network is run on a batch of search crops, one per
tracked video, each with its own template. One tracker per process: both repositories
name their package `lib`.
"""
import os
import sys

import numpy as np
import torch


class Backend(object):
    def __init__(self, name, config, finetuned=None):
        """finetuned: optional finetune_multi.py checkpoint (ORTrack): the trained attention
        weights are loaded into the network and the slot codes are added to the templates."""
        self.name = name
        self.slot = None
        self._finetuned = finetuned
        if name == 'uetrack':
            sys.path.insert(0, '/mnt/c/Users/justi/UETrack/UETrack')
            _load = torch.load
            torch.load = lambda *a, **k: _load(*a, **{'weights_only': False, **k})
            import lib.models.uetrack.encoder as encoder_module
            encoder_module.is_main_process = lambda: False
            from lib.test.parameter.uetrack import parameters
            from lib.test.tracker.uetrack import UETrack
            from lib.test.tracker.utils import sample_target, transform_image_to_crop
            from lib.utils.box_ops import clip_box
            params = parameters(config)
            params.debug = 0
            self.t = UETrack(params, 'uav')
            self._transform = transform_image_to_crop
            # text features, as UETrack.initialize() (no description on these datasets: the
            # same empty-text tokens for every video, so computed once)
            t = self.t
            if t.multi_modal_language:
                text_data, _ = t.extract_token_from_nlp_clip(None)
                with torch.no_grad():
                    t.text_src = t.network.forward_textencoder_inference(text_data=text_data.unsqueeze(0).cuda())
            else:
                t.text_src = None
        else:
            sys.path.insert(0, '/mnt/c/Users/justi/ORTrack')
            import run_dataset  # noqa: F401  (torch 2.x shims, sys.path)
            from lib.test.parameter.ortrack import parameters
            from lib.test.tracker.ortrack import ORTrack
            from lib.train.data.processing_utils import sample_target
            from lib.utils.box_ops import clip_box
            params = parameters(config)
            params.debug = 0
            self.t = ORTrack(params, 'uav')
        self.p = self.t.params
        self.sample_target, self.clip_box = sample_target, clip_box
        self.hann = self.t.output_window            # (1, 1, S, S)
        self.feat_sz = self.hann.shape[-1]
        self.merger = None
        if self._finetuned:
            assert name == 'ortrack', 'fine-tuned multi-template weights exist for ORTrack only'
            ck = torch.load(self._finetuned, map_location='cuda', weights_only=False)
            if ck.get('type') == 'merger':          # option 1d: frozen ORTrack + learned memory merger
                from finetune_multi import MemoryMerger
                self.merger = MemoryMerger(self.t.network.backbone.pos_embed_z.shape[-1], ck['max_views']).cuda()
                self.merger.load_state_dict(ck['merger'])
                self.merger.eval()
            else:                                   # option 2: fine-tuned attention + slot codes
                missing = self.t.network.load_state_dict(ck['net'], strict=False)
                assert not missing.unexpected_keys, missing.unexpected_keys
                self.slot = ck['slot'].cuda()       # (n_slots, 1, dim)

    # ------------------------------------------------------------------ template
    @torch.no_grad()
    def template(self, image, box):
        """Template of one video, as the official initialize(): an opaque dict."""
        t, p = self.t, self.p
        out = self.sample_target(image, list(box), p.template_factor, output_sz=p.template_size)
        if self.name == 'uetrack':
            z = t.preprocessor.process(out[0])
            if t.multi_modal_vision and z.size(1) == 3:
                z = torch.cat((z, z), axis=1)
            anno = self._transform(torch.tensor(list(box)), torch.tensor(list(box)), out[1],
                                   torch.Tensor([p.template_size, p.template_size]), normalize=True)
            return {'z': z, 'anno': anno.to(z.device).unsqueeze(0)}
        z = t.preprocessor.process(out[0], out[2])
        return {'z': z.tensors}

    # ------------------------------------------------------------------ search
    def crop(self, image, state, factor):
        """Search crop (numpy) and its resize factor, as the official track()."""
        out = self.sample_target(image, list(state), factor, output_sz=self.p.search_size)
        return out

    def _prep(self, crops):
        """The trackers' Preprocessor.process for a whole batch in one transfer: the
        same element-wise arithmetic ((x / 255 - mean) / std), so identical values."""
        pre = self.t.preprocessor
        x = torch.from_numpy(np.stack([c[0] for c in crops])).cuda().float().permute(0, 3, 1, 2)
        # contiguous: same values either way, but the official layout picks the same conv kernels
        return (((x / 255.0) - pre.mean) / pre.std).contiguous()

    @torch.no_grad()
    def forward(self, crops, templates):
        """Network on a batch: crops = list of sample_target outputs, templates = list of
        template dicts. Returns score (B,1,S,S), size (B,2,S,S), offset (B,2,S,S)."""
        if self.slot is not None:                   # fine-tuned network: one code path
            return self.forward_multi(crops, [[tp] for tp in templates])
        t = self.t
        if self.name == 'uetrack':
            x = self._prep(crops)
            if t.multi_modal_vision and x.size(1) == 3:
                x = torch.cat((x, x), axis=1)
            z = torch.cat([tp['z'] for tp in templates])
            anno = torch.cat([tp['anno'] for tp in templates])
            b = len(crops)
            text_src = getattr(t, 'text_src', None)
            text = None if text_src is None else text_src.expand(b, *text_src.shape[1:])
            task = None if t.task_index_batch is None else t.task_index_batch.expand(b)
            enc, _, _ = t.network.forward_encoder([z], [x], [anno], text, task)
            out = t.network.forward_decoder(feature=enc)
        else:
            x = self._prep(crops)
            z = torch.cat([tp['z'] for tp in templates])
            out = t.network.forward(template=z, search=x, is_distill=t.is_distill)
        return out['score_map'], out['size_map'], out['offset_map']

    @torch.no_grad()
    def forward_multi(self, crops, template_lists):
        """Network on a batch where every sample sees SEVERAL templates in one pass: their
        tokens are placed side by side in front of the search tokens, so attention can use
        all of them (the trackers were trained with one template). Every sample must have
        the same number of templates. With one template this equals forward()."""
        t = self.t
        n = len(template_lists[0])
        assert all(len(tl) == n for tl in template_lists)
        x = self._prep(crops)
        if self.name == 'uetrack':
            if t.multi_modal_vision and x.size(1) == 3:
                x = torch.cat((x, x), axis=1)
            zs = [torch.cat([tl[j]['z'] for tl in template_lists]) for j in range(n)]
            annos = [torch.cat([tl[j]['anno'] for tl in template_lists]) for j in range(n)]
            b = len(crops)
            text_src = getattr(t, 'text_src', None)
            text = None if text_src is None else text_src.expand(b, *text_src.shape[1:])
            task = None if t.task_index_batch is None else t.task_index_batch.expand(b)
            enc, _, _ = t.network.forward_encoder(zs, [x], annos, text, task)
            out = t.network.forward_decoder(feature=enc)
        else:
            # ORTrack's ViT.forward_features (vision_transformer.py) with n template blocks
            bb = t.network.backbone
            xt = bb.patch_embed(x) + bb.pos_embed_x
            zt = [bb.patch_embed(torch.cat([tl[j]['z'] for tl in template_lists])) + bb.pos_embed_z
                  + (self.slot[min(j, len(self.slot) - 1)] if self.slot is not None else 0.)
                  for j in range(n)]
            if self.merger is not None and len(zt) > 1:
                zt = [zt[0], self.merger(zt[1:])]   # memory views (newest first) -> one block
            h = bb.pos_drop(torch.cat(zt + [xt], dim=1))
            for blk in bb.blocks:
                h = blk(h)
            out = t.network.forward_head(bb.norm(h), None)   # reads the last search tokens
        return out['score_map'], out['size_map'], out['offset_map']

    @torch.no_grad()
    def decode(self, score, size, offset, window):
        """Boxes in crop-normalised (cx, cy, w, h) for a per-sample window strength
        (1 = official Hanning, 0 = none). Returns (B, 4) and windowed peak (B,), numpy."""
        w = torch.as_tensor(window, dtype=score.dtype, device=score.device).view(-1, 1, 1, 1)
        response = (w * self.hann + (1 - w)) * score
        if self.name == 'uetrack':
            boxes, peak = self.t.network.decoder.cal_bbox(response, size, offset, return_score=True)
        else:
            boxes = self.t.network.box_head.cal_bbox(response, size, offset)
            peak = response.flatten(1).max(1, keepdim=True)[0]
        return boxes.view(-1, 4).cpu().numpy(), peak.view(-1).cpu().numpy()

    def box_back(self, box01, state, resize_factor, image_hw):
        """Official map_box_back + clip_box for one sample."""
        p = self.p
        # the official `box * search_size / resize_factor` on a CUDA float32 tensor: torch
        # divides by a CPU scalar by multiplying with its float32 reciprocal
        inv = np.float32(1.0) / np.float32(resize_factor)
        cx, cy, w, h = ((np.asarray(box01, np.float32) * np.float32(p.search_size)) * inv).tolist()
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        half = 0.5 * p.search_size / resize_factor
        box = [cx + (cx_prev - half) - 0.5 * w, cy + (cy_prev - half) - 0.5 * h, w, h]
        return self.clip_box(box, image_hw[0], image_hw[1], margin=10)
