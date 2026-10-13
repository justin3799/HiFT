"""A frozen tracker (cfg.RL.TRACKER) with an RL policy choosing its
post-processing parameters per frame."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from pysot.core.config import cfg
from pysot.rl import backends, core
from pysot.rl.policy import ActorCritic
from pysot.tracker.hift_tracker import HiFTTracker


def load_policy(path):
    """Load a train_reinforce.py checkpoint; returns (policy, checkpoint dict)."""
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    policy = ActorCritic(core.obs_dim(), core.to_unit(core.start_action()),
                         hidden=ckpt['hidden']).cuda()
    policy.load_state_dict(ckpt['state_dict'])
    policy.eval()
    return policy, ckpt


class RLTracker(HiFTTracker):
    def __init__(self, model, policy):
        super(RLTracker, self).__init__(model)
        self.policy = policy
        self.hann = core.hanning_window()

    def init(self, img, bbox):
        # as HiFTTracker.init (ADSiamAPNTracker.init is the same), with the
        # template features taken through the backend of cfg.RL.TRACKER
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.scaleaa = s_z
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.zf = backends.template(self.model, z_crop)
        self.gate_left = 0
        self.factor = 1.0
        self.prev_a01 = core.to_unit(core.start_action())
        self.ema = 0.
        self.prev_max = 0.
        self.actions = []

    def track(self, img):
        img_hw = np.array([img.shape[:2]], dtype=np.float64)
        size = self.size[None].astype(np.float64)
        s_x, scale = core.search_region(size, np.array([self.scaleaa]), img_hw,
                                        np.array([self.factor]))
        x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                                    s_x[0], self.channel_average)
        pred, score, cls_prob, ctr_prob = core.decode(backends.track(self.model, x_crop, self.zf))
        size_crop = size * scale[:, None]
        obs = core.make_obs(pred, score, cls_prob, ctr_prob, size_crop, np.array([self.factor]),
                            size, img_hw, self.prev_a01[None], np.array([self.ema]),
                            np.array([self.prev_max]))
        u, _, _ = self.policy.act(torch.from_numpy(obs).cuda(), deterministic=True)
        a01 = ActorCritic.squash(u).cpu().numpy().astype(np.float64)
        if cfg.RL.GATE_DROP:
            # same trigger as TrackingEnv: act only after the peak score drops
            if self.ema > 0 and score.max() < cfg.RL.GATE_DROP * self.ema:
                self.gate_left = cfg.RL.GATE_HOLD
            if self.gate_left == 0:
                a01 = core.to_unit(core.baseline_action())[None]
            self.gate_left = max(self.gate_left - 1, 0)
        action = core.from_unit(a01)
        box, lr, best = core.postprocess(pred, score, self.hann, size_crop, action)
        center, size = core.update_state(self.center_pos[None], size, box, scale, lr, img_hw)
        self.center_pos, self.size = center[0], size[0]

        self.factor = action[0, 3]
        self.prev_a01 = a01[0]
        self.prev_max = best[0]
        self.ema = best[0] if self.ema == 0 else 0.9 * self.ema + 0.1 * best[0]
        self.actions.append(action[0].tolist())

        cx, cy = self.center_pos
        width, height = self.size
        return {'bbox': [cx - width / 2, cy - height / 2, width, height],
                'best_score': best[0]}
