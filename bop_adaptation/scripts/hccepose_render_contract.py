#!/usr/bin/env python3
"""Verify declared label-renderer compatibility, then invoke unchanged pose training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from hccepose_data import sha


def verify_render_contract(root):
    source=json.loads((root/'INPUTS_READY.json').read_text());policy=source.get('render_compatibility')
    if policy is None:return False  # exact legacy label recipe remains supported
    from hccepose_label_clipping import recipe
    if policy!=recipe():raise ValueError('declared renderer recipe differs')
    labels=json.loads((root/'LABELS_READY.json').read_text());extension=labels.get('renderer_extension')
    if not extension or any(extension.get(k)!=v for k,v in policy.items()):raise ValueError('missing/changed renderer attestation')
    receipt=root/'LABEL_RENDER_COMPAT.json'
    if sha(receipt)!=extension['receipt_sha256']:raise ValueError('renderer receipt changed')
    if json.loads(receipt.read_text())['policy']!=policy:raise ValueError('renderer receipt policy differs')
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--allow-nonstandard-backbone',action='store_true');a=p.parse_args()
    if not a.allow_nonstandard_backbone:p.error('explicit HCCEPose reproduction opt-out required')
    verify_render_contract(a.root)
    command=[sys.executable,'-u',str(Path(__file__).with_name('hccepose_train.py')),'train','--root='+str(a.root),'--allow-nonstandard-backbone']
    if a.resume:command.append('--resume')
    subprocess.run(command,check=True)
