"""Frozen common pilot constants, not a result or GPU dispatch."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(os.environ.get('FORKMERGE_BUDGET_PARENT',ROOT/'runtime/budget'))
PREVIOUS=Path(os.environ.get('FORKMERGE_PREVIOUS_RUN',ROOT/'runtime/previous'))
PYTHON=sys.executable
TOTAL_GPU_SECONDS=5400.
PRIOR_GPU_SECONDS=193.3471000213176
ARM_CAP_SECONDS=2100.
TOKENS=64
WIDTH=1536
SIZE=(448,256)
FRAMES=21
FPS=4
STEPS=16
SHIFT=16.
CFG=5.
SEEDS=(20261021,20261022)
UPDATES=32
LR=3e-4
PAIR_COEFFICIENT=.1
NEW_COEFFICIENT=.1
METHODS=('paired','paired_worst_witness')
SCENARIOS=('cup','teapot','candy')
ACTIONS={'observe':0,'interact':1,'replace_cup':2,'wash_refill_pot':3,'replace_lid':4,'empty_refill_box':5,'occlude':6}
PROMPTS={
 'cup':'Locked camera, continuous tabletop shot. Show the replacement cup, then move the freestanding screen aside to reveal the tablecloth on the right. Preserve object identity and the existing scene. No cuts.',
 'teapot':'Locked camera, continuous tabletop shot. Pour a small amount from the teapot into the clean display glass on the left. Then move the freestanding screen aside to reveal the receiver glass on the right. Preserve object identity and the existing scene. No cuts.',
 'candy':'Locked camera, continuous tabletop shot. Tilt the box to reveal its contents without spilling. Then move the freestanding screen aside to reveal the objects on the right. Preserve object identity and the existing scene. No cuts.'}
NEGATIVE='Camera motion, cuts, text overlays, extra objects, duplicated objects, changing background.'
