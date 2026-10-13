"""Template-MEMORY environment on a frozen UETrack / ORTrack (batched over videos).

Generalises tmplenv.py from one extra template to a memory of --slots extra templates. The
frame-0 template is always kept as an anchor; all stored views are fed to the network in one
pass together with it (Backend.forward_multi: template tokens side by side). Every --interval
frames the policy chooses:

  0 keep    leave the memory unchanged
  1 store   store a crop around the current box (last tracked frame) as a new view; when the
            memory is full the OLDEST view is replaced
  2 clear   empty the memory (back to the frame-0 template only)

Reward (training): each policy tracker has a paired official tracker (frame-0 template only)
started from the same frame; the reward of a decision is the summed IoU difference over the
following interval. Evaluation tracks whole videos.

Observation: tracking state (peaks, running average, score-map statistics, motion, size,
border, aspect), and per memory slot: filled, age, similarity of the current target crop
to the stored view; plus similarity to the anchor, number of stores, previous action.
"""
import cv2
import numpy as np
import torch

from frenv import iou, success_auc
from tmplenv import ncc

ACTIONS = ['keep', 'store', 'clear']
N_BASE = 16          # tracking-state features + anchor similarity + #stores


def obs_dim(slots):
    return N_BASE + 3 * slots + len(ACTIONS)


class MemEnv(object):
    def __init__(self, backend, videos, num_envs, slots=3, episode_len=400, interval=10, train=True,
                 paired=True, seed=0, init_jitter=0.0):
        self.b, self.videos, self.n, self.slots = backend, videos, num_envs, slots
        self.k = 2 * num_envs if paired else num_envs
        self.paired, self.train = paired, train
        self.episode_len, self.interval, self.init_jitter = episode_len, interval, init_jitter
        self.rng = np.random.RandomState(seed)
        self.queue = list(range(len(videos)))
        self.obs_dim = obs_dim(slots)
        k = self.k
        self.active = np.zeros(k, bool)
        self.video, self.t, self.end = np.zeros(k, int), np.zeros(k, int), np.zeros(k, int)
        self.state, self.prev_state = np.zeros((k, 4)), np.zeros((k, 4))
        self.z0, self.c0 = [None] * k, [None] * k
        self.mem = [[] for _ in range(k)]                  # list of (template, crop, birth frame)
        self.last_img = [None] * k
        self.boxes, self.acts = [None] * k, [None] * k
        self.ema, self.peaks = np.zeros(k), np.zeros((k, 3))
        self.stats = np.zeros((k, 4))
        self.prev_a = np.zeros(k, int)
        self.stores = np.zeros(k)

    # ------------------------------------------------------------------ helpers
    def _img(self, v, t, memo):
        if (v, t) not in memo:
            memo[(v, t)] = cv2.cvtColor(cv2.imread(self.videos[v]['frames'][t]), cv2.COLOR_BGR2RGB)
        return memo[(v, t)]

    def _crop_tmpl(self, img, box):
        p = self.b.p
        return self.b.sample_target(img, list(box), p.template_factor, output_sz=p.template_size)[0]

    def _start(self, i, memo):
        if self.train:
            v = self.rng.randint(len(self.videos))
            gt = self.videos[v]['gt']
            ok = np.where((gt[:-1, 2] > 0) & (gt[:-1, 3] > 0))[0]
            if not len(ok):
                return self._start(i, memo)
            s = int(self.rng.choice(ok))
            e = min(len(gt) - 1, s + self.episode_len)
        else:
            if not self.queue:
                return False
            v, s = self.queue.pop(0), 0
            e = len(self.videos[v]['gt']) - 1
        box0 = self.videos[v]['gt'][s].copy()
        if self.init_jitter:
            box0[:2] += self.rng.normal(0, self.init_jitter * np.sqrt(max(box0[2] * box0[3], 1.)), 2)
        img = self._img(v, s, memo)
        z0, c0 = self.b.template(img, box0), self._crop_tmpl(img, box0)
        for j in ([i, i + self.n] if self.paired else [i]):
            self.active[j] = True
            self.video[j], self.t[j], self.end[j] = v, s + 1, e
            self.state[j] = self.prev_state[j] = box0
            self.z0[j], self.c0[j], self.mem[j] = z0, c0, []
            self.prev_a[j], self.stores[j] = 0, 0
            self.last_img[j] = img
            self.boxes[j], self.acts[j] = [box0.tolist()], []
            self.ema[j], self.peaks[j], self.stats[j] = 0., 0., 0.
        return True

    def reset(self):
        memo = {}
        self.active[:] = False
        for i in range(self.n):
            if not self._start(i, memo):
                break
        return self._observe()

    def policy_slots(self):
        return np.where(self.active[:self.n])[0]

    # ------------------------------------------------------------------ one tracking frame
    def _track_frame(self, ids, memo):
        b = self.b
        imgs = [self._img(self.video[j], self.t[j], memo) for j in ids]
        crops = [b.crop(im, self.state[j], b.p.search_factor) for im, j in zip(imgs, ids)]
        tl = [[self.z0[j]] + [m[0] for m in self.mem[j]] for j in ids]
        score = size = offset = None
        for n in sorted({len(x) for x in tl}):
            rows = [r for r in range(len(ids)) if len(tl[r]) == n]
            s, z, o = b.forward_multi([crops[r] for r in rows], [tl[r] for r in rows])
            if score is None:
                score = s.new_zeros((len(ids),) + tuple(s.shape[1:]))
                size = z.new_zeros((len(ids),) + tuple(z.shape[1:]))
                offset = o.new_zeros((len(ids),) + tuple(o.shape[1:]))
            idx = torch.as_tensor(rows, device=s.device)
            score[idx], size[idx], offset[idx] = s, z, o
        b01, peak = b.decode(score, size, offset, np.ones(len(ids)))
        stats = self._map_stats(score)
        ov = np.zeros(len(ids))
        for r, j in enumerate(ids):
            new = np.asarray(b.box_back(b01[r], self.state[j], crops[r][1], imgs[r].shape[:2]), dtype=np.float64)
            self.prev_state[j], self.state[j] = self.state[j].copy(), new
            self.boxes[j].append(new.tolist())
            self.last_img[j] = imgs[r]
            self.ema[j] = peak[r] if self.ema[j] == 0 else 0.9 * self.ema[j] + 0.1 * peak[r]
            self.peaks[j] = [peak[r], self.peaks[j, 0], self.peaks[j, 1]]
            self.stats[j] = stats[r]
            gt = self.videos[self.video[j]]['gt'][self.t[j]]
            ov[r] = iou(new[None], gt[None])[0]
            self.t[j] += 1
        return ov

    @staticmethod
    def _map_stats(score):
        S = score.shape[-1]
        s = score.flatten(1)
        smax, arg = s.max(1)
        ys, xs = (arg // S).float(), (arg % S).float()
        yy, xx = torch.meshgrid(torch.arange(S, device=s.device), torch.arange(S, device=s.device), indexing='ij')
        far = ((yy.flatten()[None] - ys[:, None]).abs() > 2) | ((xx.flatten()[None] - xs[:, None]).abs() > 2)
        second = (s * far).max(1)[0] / smax.clamp_min(1e-6)
        p = s / s.sum(1, keepdim=True).clamp_min(1e-6)
        ent = -(p * p.clamp_min(1e-12).log()).sum(1) / np.log(S * S)
        ctr = ((ys - (S - 1) / 2) ** 2 + (xs - (S - 1) / 2) ** 2).sqrt() / (S / 2)
        return torch.stack([smax, second, ent, ctr], 1).cpu().numpy()

    # ------------------------------------------------------------------ observation
    def _base(self, j):
        """N_BASE tracking-state features of tracker j and the crop around its current box."""
        H, W = self.last_img[j].shape[:2]
        st = self.state[j]
        w, h = max(st[2], 1.), max(st[3], 1.)
        cand = self._crop_tmpl(self.last_img[j], st)
        d = np.hypot(*(st[:2] + st[2:] / 2 - self.prev_state[j, :2] - self.prev_state[j, 2:] / 2)) / np.sqrt(w * h)
        cx, cy = st[0] + w / 2, st[1] + h / 2
        base = [self.peaks[j, 0], self.peaks[j, 1], self.peaks[j, 2], self.ema[j],
                self.peaks[j, 0] / max(self.ema[j], 1e-6), *self.stats[j],
                min(d, 5.), np.log(w / max(self.prev_state[j, 2], 1.)),
                np.sqrt(w * h / (H * W)), min(min(cx, cy, W - cx, H - cy) / np.sqrt(w * h), 5.), np.log(w / h),
                ncc(cand, self.c0[j]), min(self.stores[j] / 20., 1.)]
        return base, cand

    def _observe(self):
        ids = self.policy_slots()
        out = np.zeros((len(ids), self.obs_dim), np.float32)
        for r, j in enumerate(ids):
            base, cand = self._base(j)
            slot = []
            for q in range(self.slots):             # newest first
                if q < len(self.mem[j]):
                    m = self.mem[j][-1 - q]
                    slot += [1., min((self.t[j] - m[2]) / 300., 1.), ncc(cand, m[1])]
                else:
                    slot += [0., 0., 0.]
            out[r, :N_BASE + 3 * self.slots] = base + slot
            out[r, N_BASE + 3 * self.slots + self.prev_a[j]] = 1.
        return out

    # ------------------------------------------------------------------ step
    def step(self, actions):
        self._apply(actions)
        return self._advance()

    def _apply(self, actions):
        for a, i in zip(actions, self.policy_slots()):
            if a == 1:
                view = (self.b.template(self.last_img[i], self.state[i]),
                        self._crop_tmpl(self.last_img[i], self.state[i]), self.t[i] - 1)
                self.mem[i].append(view)
                if len(self.mem[i]) > self.slots:
                    self.mem[i].pop(0)                  # replace the oldest view
                self.stores[i] += 1
            elif a == 2:
                self.mem[i] = []
            self.prev_a[i] = a
            self.acts[i].append(int(a))

    def _advance(self):
        """Track `interval` frames with the current template sets; reward, episode ends."""
        reward = np.zeros(self.n)
        done = np.zeros(self.n, bool)
        finished = []
        for _ in range(self.interval):
            ids = np.where(self.active & (self.t <= self.end))[0]
            if not len(ids):
                break
            ov = self._track_frame(ids, {})
            if self.paired:
                pos = {j: r for r, j in enumerate(ids)}
                for r, j in enumerate(ids):
                    if j < self.n and j + self.n in pos:
                        reward[j] += ov[r] - ov[pos[j + self.n]]
        memo = {}
        for i in range(self.n):
            if not self.active[i] or self.t[i] <= self.end[i]:
                continue
            done[i] = True
            if not self.train:
                v = self.videos[self.video[i]]
                rec = {'name': v['name'], 'boxes': self.boxes[i], 'actions': self.acts[i],
                       'auc': success_auc(self.boxes[i], v['gt'])}
                if self.paired:
                    rec['official_boxes'] = self.boxes[i + self.n]
                    rec['official_auc'] = success_auc(self.boxes[i + self.n], v['gt'])
                finished.append(rec)
            for j in ([i, i + self.n] if self.paired else [i]):
                self.active[j] = False
            self._start(i, memo)
        return self._observe(), reward, done, finished


# ---------------------------------------------------------------------------- design (b)
FIXED_ACTIONS = ['off', 'on']


def obs_dim_fixed(n_lags):
    return N_BASE + n_lags + 1 + len(FIXED_ACTIONS)


class FixedMemEnv(MemEnv):
    """Design (b): the memory always consists of the views at fixed lags back in time
    (default t-15, t-30, t-60, cut at the tracker's own boxes of those frames). Every
    `interval` frames the policy only switches the memory off (frame-0 template only, the
    official tracker) or on (frame-0 template + the lagged views, re-cut at each decision so
    they slide along with the video). Observation: tracking state, similarity of the current
    target crop to each lagged view, whether the memory is on, previous action."""

    def __init__(self, backend, videos, num_envs, lags=(15, 30, 60), **kw):
        super(FixedMemEnv, self).__init__(backend, videos, num_envs, slots=len(lags), **kw)
        self.lags = list(lags)
        self.obs_dim = obs_dim_fixed(len(self.lags))
        self.s0 = np.zeros(self.k, int)
        self.on = np.zeros(self.k, bool)

    def _start(self, i, memo):
        ok = super(FixedMemEnv, self)._start(i, memo)
        if ok:
            for j in ([i, i + self.n] if self.paired else [i]):
                self.s0[j], self.on[j] = self.t[j] - 1, False
        return ok

    def _views(self, j, memo):
        """(template, crop) of the lagged frames of tracker j, own boxes (clamped to the
        episode start)."""
        last = self.t[j] - 1
        out = []
        for lag in self.lags:
            f = max(self.s0[j], last - lag)
            img = self._img(self.video[j], f, memo)
            box = self.boxes[j][f - self.s0[j]]
            out.append((self.b.template(img, box), self._crop_tmpl(img, box), f))
        return out

    def _observe(self):
        ids = self.policy_slots()
        out = np.zeros((len(ids), self.obs_dim), np.float32)
        memo = {}
        self._cache = {}
        for r, j in enumerate(ids):
            base, cand = self._base(j)
            views = self._views(j, memo)
            self._cache[j] = views
            sims = [ncc(cand, v[1]) for v in views]
            out[r, :N_BASE + len(self.lags) + 1] = base + sims + [float(self.on[j])]
            out[r, N_BASE + len(self.lags) + 1 + self.prev_a[j]] = 1.
        return out

    def _apply(self, actions):
        for a, i in zip(actions, self.policy_slots()):
            if a == 1:
                views = self._cache.get(i) or self._views(i, {})
                self.mem[i] = list(views)
                if not self.on[i]:
                    self.stores[i] += 1
                self.on[i] = True
            else:
                self.mem[i] = []
                self.on[i] = False
            self.prev_a[i] = a
            self.acts[i].append(int(a))
