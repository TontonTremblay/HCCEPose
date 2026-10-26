"""Verify the explicit, SHA-bound geometry review before using the compatibility path."""
import json
from pathlib import Path
from hccepose_data import sha


def verify_visibility_policy(policy):
    scripts=Path(__file__).parent
    expected={'checker_sha256':scripts/'hccepose_prepare_visibility.py',
              'clipping_extension_sha256':scripts/'hccepose_label_clipping.py',
              'frozen_adapter_sha256':scripts/'hccepose_data.py'}
    for key,path in expected.items():
        if sha(path)!=policy[key]:raise ValueError('reviewed implementation changed: '+key)
    root=Path(policy['review_root']);approval=root/'APPROVED.json'
    if sha(approval)!=policy['approval_sha256']:raise ValueError('geometry review approval changed')
    proof=json.loads(approval.read_text())
    if proof.get('approved') is not True or not proof.get('normal_maps_byte_identical'):raise ValueError('geometry review incomplete')
    for key in expected:
        if proof[key]!=policy[key]:raise ValueError('approval implementation identity differs')
    for filename,key in [('NUMERICAL_REVIEW.json','numerical_review_sha256'),('FOCUS.json','focus_sha256'),
                         ('CLIPPING_REVIEW.json','clipping_review_sha256'),('PARITY.json','parity_sha256'),('INPUTS_READY.json','input_sha256')]:
        if sha(root/filename)!=proof[key]:raise ValueError('review evidence changed: '+filename)
    if json.loads((root/'NUMERICAL_REVIEW.json').read_text()).get('passed') is not True:raise ValueError('numerical geometry review failed')
    for name,digest in proof['images'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(root/path)!=digest:raise ValueError('reviewed overlay changed')
    return proof
