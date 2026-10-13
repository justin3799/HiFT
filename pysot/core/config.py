# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "HiFT_alexnet"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()


__C.TRAIN.THR_HIGH = 0.6

__C.TRAIN.clsandlocchannel = 256

__C.TRAIN.groupchannel = 32

__C.TRAIN.THR_LOW = 0.3

__C.TRAIN.NEG_NUM = 16

__C.TRAIN.POS_NUM = 16

__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.PR = 1

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 3.0

__C.TRAIN.SHAPE_WEIGHT =2.0

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 287 #255

__C.TRAIN.OUTPUTFEATURE_SIZE = 11

__C.TRAIN.LABEL_RANGE = 4

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 21 #25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = 1

__C.TRAIN.LARGER=2.0

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 30

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 100

__C.TRAIN.NUM_GPU = 2

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.w1=1.0

__C.TRAIN.w2=1.0

__C.TRAIN.w3=1.0

__C.TRAIN.w4=1.0

__C.TRAIN.w5=1.0

__C.TRAIN.range=2.0


__C.TRAIN.MASK_WEIGHT = 1

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.2  

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'GOT', 'YOUTUBEBB')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1


__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = 'training_dataset/got10k/crop511'         # GOT dataset path
__C.DATASET.GOT.ANNO = 'training_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 100000

# small UAV123-based set built by tools/make_short_data.py, for debugging
__C.DATASET.SHORT = CN()
__C.DATASET.SHORT.ROOT = 'short_train/crop511'
__C.DATASET.SHORT.ANNO = 'short_train/train.json'
__C.DATASET.SHORT.FRAME_RANGE = 50
__C.DATASET.SHORT.NUM_USE = -1

__C.DATASET.VIDEOS_PER_EPOCH = 600000



# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'alexnet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = 'back.pth'

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer3', 'layer4', 'layer5']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# # ------------------------------------------------------------------------ #
# # Anchor options
# # ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# # Anchor stride
__C.ANCHOR.STRIDE = 16


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'HiFTtracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

__C.TRACK.w1=1.2 

__C.TRACK.w2=1.0

__C.TRACK.w3=1.6 

__C.TRACK.LARGER=1.4
# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

__C.TRACK.STRIDE = 8 

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127

# ------------------------------------------------------------------------ #
# Reinforcement learning options (tools/train_reinforce.py)
# A policy sets the tracker's post-processing parameters every frame while
# the HiFT network stays frozen.
# ------------------------------------------------------------------------ #
__C.RL = CN()

# Frozen tracker the policy is trained on top of: 'HiFT' or 'SiamAPN++'
__C.RL.TRACKER = 'HiFT'

# Its weights
__C.RL.SNAPSHOT = './snapshot/first.pth'

# Dataset the training videos are drawn from
__C.RL.DATASET = 'UAV123'

__C.RL.DATASET_DIR = '../testing_dataset'

# Separate training / validation sets inside RL.DATASET (e.g. GOT-10k's
# 'GOT10k_train.json' / 'GOT10k_val.json' from tools/gen_got10k_json.py);
# empty: cross-validation folds of RL.DATASET as below
__C.RL.TRAIN_JSON = ''

__C.RL.VAL_JSON = ''

# Validate on this many randomly chosen validation videos (0: all)
__C.RL.VAL_VIDEOS = 0

# Cross-validation: videos are grouped by source video (car6_1..car6_5 -> car6)
# and split into NUM_FOLDS folds; the policy trains on every fold except FOLD
__C.RL.NUM_FOLDS = 2

__C.RL.FOLD = 0

# Fraction of the training videos kept aside to pick the best checkpoint
__C.RL.VAL_FRACTION = 0.2

__C.RL.SPLIT_SEED = 0

# Action ranges: [window influence, penalty k, size lr, search scale]
__C.RL.ACTION_LOW = [0.0, 0.0, 0.0, 0.8]

__C.RL.ACTION_HIGH = [0.9, 0.3, 1.0, 2.0]

# Parallel tracking episodes, each on its own video
__C.RL.NUM_ENVS = 16

# 'iou': reward is the frame's IoU with ground truth.
# 'relative': IoU minus that of a hand-tuned HiFT tracker run alongside on the
# same frames (paired counterfactual baseline; costs a second tracker per env)
__C.RL.REWARD = 'iou'

# Subtracted from the reward each frame, times the squared distance (unit
# action space) between the action and the hand-tuned constants: the policy
# only departs from them when that pays off
__C.RL.DEVIATION_COEF = 0.0

# Action the policy starts from, in unit action space ([0, 1] per dimension).
# Empty: the tracker's hand-tuned constants. Set (e.g. drawn at random) to
# train without any knowledge of the constants: it then also replaces them as
# the shadow tracker in the relative reward, the anchor of the deviation cost,
# the first observed previous action, and the bar for model selection.
__C.RL.INIT_ACTION = []

# Paired reward: every SHADOW_SYNC frames, reset the shadow tracker to the
# policy tracker's state, so the reward measures the local effect of recent
# decisions rather than a long, chaotic divergence (0: never)
__C.RL.SHADOW_SYNC = 0

# Event-triggered control: the policy acts only after the peak score falls
# below GATE_DROP times its running average, for GATE_HOLD frames; elsewhere
# the hand-tuned constants are used (0: act on every frame)
__C.RL.GATE_DROP = 0.0

__C.RL.GATE_HOLD = 30

# Include the two 11x11 score maps in the state (else summary statistics only)
__C.RL.OBS_MAPS = True

# Validation: average over VAL_DRAWS runs with actions jittered by VAL_JITTER
# (unit space, same for baseline and policy), and only save a checkpoint that
# beats the best so far by VAL_MARGIN; the hand-tuned constants are the start
__C.RL.VAL_DRAWS = 1

__C.RL.VAL_JITTER = 0.0

__C.RL.VAL_MARGIN = 0.0

# Frames per episode (shorter if the video ends first)
__C.RL.EPISODE_LEN = 300

# Steps per env collected before each PPO update
__C.RL.ROLLOUT_LEN = 128

__C.RL.UPDATES = 300

__C.RL.PPO_EPOCHS = 4

__C.RL.MINIBATCH = 512

__C.RL.LR = 3e-4

__C.RL.GAMMA = 0.95

__C.RL.GAE_LAMBDA = 0.95

__C.RL.CLIP = 0.2

__C.RL.VALUE_COEF = 0.5

__C.RL.ENTROPY_COEF = 0.001

__C.RL.GRAD_CLIP = 0.5

# Initial std of the Gaussian policy (pre-squash)
__C.RL.INIT_LOG_STD = -1.0

__C.RL.HIDDEN = 256

# Validate (and save the best checkpoint) every VAL_EVERY updates
__C.RL.VAL_EVERY = 25

__C.RL.LOG_DIR = './logs_rl'

__C.RL.SNAPSHOT_DIR = './snapshot_rl'
