#!/usr/bin/env python3
"""Portable manual-review and GPU ABI contract for the fixed BF training recipe.

The contract validates synthetic PnP, adaptive-loss serialization and checkpoint
IO. Smoke weights are used only for this probe, never to initialize full training.
"""
import argparse
import json
from pathlib import Path
from hccepose_paths import UPSTREAM
from hccepose_data import sha,save,FrozenBOP,backend


def verify_smoke(root):
    identity=json.loads((root/'SMOKE_IDENTITY.json').read_text())
    ready=json.loads((root/'INPUTS_READY.json').read_text())
    passed=json.loads((root/'SMOKE_PASSED.json').read_text())
    if not ready['smoke_only'] or not passed['passed'] or passed['successful_updates']<20:
        raise ValueError('separate GPU smoke with at least20 actual updates required')
    if identity['inputs_sha256']!=sha(root/'INPUTS_READY.json') or identity['adapter_sha256']!=sha(Path(__file__).with_name('hccepose_data.py')):
        raise ValueError('smoke input/adapter identity changed')
    if identity['driver_sha256']!=sha(Path(__file__).with_name('hccepose_smoke.py')):raise ValueError('smoke driver changed')
    if passed['checkpoint_sha256']!=sha(root/'smoke_training/last.pth'):raise ValueError('smoke checkpoint changed')
    for name,digest in identity['upstream_files'].items():
        if sha(UPSTREAM/name)!=digest:raise ValueError('upstream source changed')
    from hccepose_label_clipping import recipe
    if ready.get('render_compatibility')!=recipe():raise ValueError('render recipe changed')
    return ready


def review(root,reviewer,acknowledged):
    if not acknowledged or not reviewer or not reviewer.strip():raise ValueError('explicit operator review acknowledgment and reviewer required')
    verify_smoke(root);images=sorted((root/'review').glob('*.jpg'))
    if not images:raise ValueError('no overlays to review')
    out=root/'REVIEWED.json'
    if out.exists():raise FileExistsError('review receipt already exists')
    save(out,{'reviewer':reviewer,'operator_declares_overlays_inspected':True,
              'not_automated_visual_approval':True,'smoke_identity_sha256':sha(root/'SMOKE_IDENTITY.json'),
              'smoke_receipt_sha256':sha(root/'SMOKE_PASSED.json'),
              'overlays_sha256':{str(p.relative_to(root)):sha(p) for p in images}})


def contract(root,experiment):
    ready=verify_smoke(root);approval=json.loads((root/'REVIEWED.json').read_text())
    if not approval.get('operator_declares_overlays_inspected') or not approval.get('reviewer'):raise ValueError('manual review required')
    if approval['smoke_identity_sha256']!=sha(root/'SMOKE_IDENTITY.json') or approval['smoke_receipt_sha256']!=sha(root/'SMOKE_PASSED.json'):raise ValueError('reviewed evidence changed')
    for name,digest in approval['overlays_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('reviewed overlay changed')
    import numpy as np
    import torch
    from HccePose.network_model import HccePose_BF_Net,HccePose_Loss
    from bop_toolkit.bop_toolkit_lib.inout import load_ply
    from hccepose_train import validate,loss_state,restore_loss,atomic_checkpoint
    torch.set_num_threads(4);obj=ready['object'];data=FrozenBOP(root);info=data.model_info[str(obj)]
    minimum=torch.tensor([info['min_'+s] for s in 'xyz'],device='cuda');size=torch.tensor([info['size_'+s] for s in 'xyz'],device='cuda')
    model=HccePose_BF_Net(efficientnet_key=None,input_channels=3,min_xyz=minimum,size_xyz=size).cuda()
    state=torch.load(root/'smoke_training/last.pth',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict'])
    val=backend().test_bop_dataset_back_front(data,'val_pbr',padding_ratio=1.5,crop_size_img=256,ratio=1.);val.update_obj_id(obj,data.obj_model_list[0])
    scores,report=validate(model,torch.utils.data.DataLoader(val,batch_size=4,num_workers=0),load_ply(data.obj_model_list[0]),info)
    criterion=HccePose_Loss();x=torch.rand(2,24,128,128,device='cuda');mask=torch.rand(2,1,128,128,device='cuda');gt=torch.rand_like(x,dtype=torch.float64)
    criterion(x,x,mask,gt,gt,mask[:,0].double());saved=loss_state(criterion);restored=HccePose_Loss();restore_loss(restored,saved,'cuda')
    a=criterion(x,x,mask,gt,gt,mask[:,0].double());b=restored(x,x,mask,gt,gt,mask[:,0].double())
    for key in ['Front_L1Losses','Back_L1Losses','mask_loss']:
        if not torch.equal(a[key],b[key]):raise ValueError('adaptive loss restoration differs')
    output=experiment/'contract'/ready['dataset'];output.mkdir(parents=True,exist_ok=True)
    if (output/'PASSED.json').exists():raise FileExistsError('contract already exists; do not overwrite')
    atomic_checkpoint(output/'checkpoint_contract.pth',{'criterion':saved,'keypoints_':scores.tolist(),'model_state_dict':model.state_dict()})
    loaded=torch.load(output/'checkpoint_contract.pth',map_location='cpu',weights_only=False)
    if loaded['keypoints_']!=scores.tolist():raise ValueError('checkpoint serialization failed')
    save(output/'PASSED.json',{'passed':True,'scope':'synthetic PnP, adaptive loss restoration and checkpoint IO; not final accuracy',
         'synthetic_smoke_validation':report,'train_driver_sha256':sha(Path(__file__).with_name('hccepose_train.py')),
         'adapter_sha256':sha(Path(__file__).with_name('hccepose_data.py')),'review_receipt_sha256':sha(root/'REVIEWED.json')})
    print('HCCE_PRODUCTION_CONTRACT_PASSED',ready['dataset'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['review','contract']);p.add_argument('--smoke-root',type=Path,required=True);p.add_argument('--experiment-root',type=Path);p.add_argument('--reviewer');p.add_argument('--acknowledge-reviewed-overlays',action='store_true');a=p.parse_args()
    if a.mode=='review':review(a.smoke_root,a.reviewer,a.acknowledge_reviewed_overlays)
    else:
        if a.experiment_root is None:p.error('--experiment-root required for contract')
        contract(a.smoke_root,a.experiment_root)
