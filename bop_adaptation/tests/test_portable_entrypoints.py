import json
from pathlib import Path
import pytest
import hccepose_contract as contract
from hccepose_paths import UPSTREAM
from hccepose_data import sha
from prepare_frozen import inventory


def test_checkout_discovery_is_not_a_cluster_installation_path():
    assert (UPSTREAM/'HccePose/network_model.py').is_file()
    assert UPSTREAM==Path(__file__).resolve().parents[2]


def test_inventory_pins_source_metadata_without_modification(tmp_path):
    (tmp_path/'geometry').mkdir();keypoints=tmp_path/'geometry/keypoints_3d.json';keypoints.write_text('{"keypoints_3d":[[0,0,0]]}')
    for split in ['train','val']:
        folder=tmp_path/'dataset'/split;folder.mkdir(parents=True)
        (folder/'annotations.json').write_text(json.dumps({'images':[{'id':1}],'annotations':[{'id':1,'image_id':1}]}))
    record=inventory(tmp_path,'example',1)
    assert record['object']==1 and record['dataset']=='example'
    assert record['keypoints_sha256']==sha(keypoints)
    for split in ['train','val']:
        assert record['splits'][split]['images']==1
        assert record['splits'][split]['sha256']==sha(tmp_path/'dataset'/split/'annotations.json')


def test_manual_review_is_explicit_and_binds_overlay_content(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='explicit'):contract.review(tmp_path,'operator',False)
    monkeypatch.setattr(contract,'verify_smoke',lambda root:{'dataset':'example','object':1})
    for name in ['SMOKE_IDENTITY.json','SMOKE_PASSED.json']:(tmp_path/name).write_text('{}')
    (tmp_path/'review').mkdir();image=tmp_path/'review/fixture.jpg';image.write_bytes(b'unit test fixture')
    contract.review(tmp_path,'unit-test-operator',True)
    proof=json.loads((tmp_path/'REVIEWED.json').read_text());assert proof['not_automated_visual_approval']
    assert proof['overlays_sha256']['review/fixture.jpg']==sha(image)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='overlay changed'):contract.contract(tmp_path,tmp_path/'experiment')
