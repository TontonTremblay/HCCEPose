"""Locate the upstream checkout without cluster-specific installation paths."""
import os
from pathlib import Path
import sys

UPSTREAM = Path(os.environ.get('HCCEPOSE_ROOT', Path(__file__).resolve().parents[2])).resolve()
if not (UPSTREAM / 'HccePose').is_dir():
    raise RuntimeError('HCCEPOSE_ROOT must point to a HCCEPose checkout')
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / 'bop_toolkit'))
