import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from hccepose_data import sha
import hccepose_label_clipping as clipping
from hccepose_render_contract import verify_render_contract
from hccepose_visibility_gate import verify_visibility_policy


def dump(path,value):path.write_text(json.dumps(value))


def test_render_attestation_required_without_invalidating_legacy_roots(tmp_path,monkeypatch):
    dump(tmp_path/'INPUTS_READY.json',{})
    assert verify_render_contract(tmp_path) is False
    policy={'extension_sha256':'reviewed','policy':'conditional'}
    monkeypatch.setattr(clipping,'recipe',lambda:policy)
    dump(tmp_path/'INPUTS_READY.json',{'render_compatibility':policy})
    dump(tmp_path/'LABELS_READY.json',{})
    with pytest.raises(ValueError,match='attestation'):verify_render_contract(tmp_path)
    dump(tmp_path/'LABEL_RENDER_COMPAT.json',{'policy':policy})
    dump(tmp_path/'LABELS_READY.json',{'renderer_extension':{**policy,'receipt_sha256':sha(tmp_path/'LABEL_RENDER_COMPAT.json')}})
    assert verify_render_contract(tmp_path)
    dump(tmp_path/'LABEL_RENDER_COMPAT.json',{'policy':'tampered'})
    with pytest.raises(ValueError,match='receipt changed'):verify_render_contract(tmp_path)


def test_review_approval_binds_code_evidence_and_actual_viewed_images(tmp_path):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    policy={'review_root':str(tmp_path),'checker_sha256':sha(scripts/'hccepose_prepare_visibility.py'),
            'clipping_extension_sha256':sha(scripts/'hccepose_label_clipping.py'),'frozen_adapter_sha256':sha(scripts/'hccepose_data.py')}
    proof={k:v for k,v in policy.items() if k!='review_root'};proof.update(approved=True,normal_maps_byte_identical=True)
    for filename,key in [('NUMERICAL_REVIEW.json','numerical_review_sha256'),('FOCUS.json','focus_sha256'),('CLIPPING_REVIEW.json','clipping_review_sha256'),('PARITY.json','parity_sha256'),('INPUTS_READY.json','input_sha256')]:
        dump(tmp_path/filename,{'passed':True});proof[key]=sha(tmp_path/filename)
    image=tmp_path/'review.jpg';image.write_bytes(b'fixture');proof['images']={'review.jpg':sha(image)}
    dump(tmp_path/'APPROVED.json',proof);policy['approval_sha256']=sha(tmp_path/'APPROVED.json')
    assert verify_visibility_policy(policy)['approved']
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='overlay'):verify_visibility_policy(policy)
