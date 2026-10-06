"""The frozen tracker networks the RL policy can sit on (cfg.RL.TRACKER)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from pysot.core.config import cfg
from pysot.utils.model_load import load_pretrain


def build_model(snapshot):
    """Frozen network in eval mode, weights from snapshot."""
    if cfg.RL.TRACKER == 'SiamAPN++':
        from pysot.models.siamapnpp import SiamAPNpp
        model = SiamAPNpp()
        state = torch.load(snapshot, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        missing, _ = model.load_state_dict(state, strict=False)
        assert not missing, 'weights missing for {}'.format(missing)
    else:
        from pysot.models.model_builder import ModelBuilder
        model = load_pretrain(ModelBuilder(), snapshot)
    # requires_grad is left as built: setting it False makes cuDNN pick other
    # kernels (outputs change by ~1e-4), so tracking would no longer match
    # test.py; forward passes run under no_grad either way
    return model.cuda().eval()


@torch.no_grad()
def template(model, z):
    """Template features (a list of tensors, batch first)."""
    return list(model.backbone(z))


@torch.no_grad()
def track(model, x, zf):
    """Raw head outputs for a batch of search crops, decoded by core.decode."""
    xf = model.backbone(x)
    if cfg.RL.TRACKER == 'SiamAPN++':
        return model.forward_head(xf, zf)
    loc, cls1, cls2 = model.grader(xf, zf)
    return {'loc': loc, 'cls1': cls1, 'cls2': cls2}
