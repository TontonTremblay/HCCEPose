#!/usr/bin/env python3
"""Bounded HCCEPose(BF) training with frozen base data and genuine synthetic val.

Upstream ResNet34, HCCE encoder/loss, augmentations, crop256/pad1.5, Adam2e-4,
batch24 and AMP retained. Final scheduled endpoint, atomic recovery checkpoints,
strict data identity, no zero-label fallback or training-set 'validation'.
"""
import argparse
import copy
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from hccepose_paths import UPSTREAM
from hccepose_data import FrozenBOP,backend,make_renderer,sha,save


def atomic_checkpoint(path,record):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        torch.save(record,stream);stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)


def label_identity(root):
    return {'input_sha256':sha(root/'INPUTS_READY.json'),'adapter_sha256':sha(Path(__file__).with_name('hccepose_data.py')),
            'driver_sha256':sha(__file__),'upstream_loader_sha256':sha(UPSTREAM/'HccePose/bop_loader.py')}


def verify_inputs(root):
    ready=json.loads((root/'INPUTS_READY.json').read_text())
    if ready['smoke_only']:raise ValueError('full training cannot use smoke data')
    for split,digest in ready['manifests_sha256'].items():
        if sha(root/(split+'.json'))!=digest:raise ValueError('input manifest changed')
    if sha(root/'models/models_info.json')!=ready['model_info_sha256']:raise ValueError('encoding bounds changed')
    if sha(root/f'models/obj_{ready["object"]:06d}.ply')!=ready['mesh_sha256']:raise ValueError('original CAD changed')
    return ready


def labels(root):
    cv2.setNumThreads(1)
    from OpenGL import GL
    ready=verify_inputs(root);obj=ready['object'];data=FrozenBOP(root);identity=label_identity(root)
    identity_path=root/'LABEL_IDENTITY.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('label algorithm/input changed; new cache required')
    save(identity_path,identity)
    records=[];completed={};journal=root/'LABEL_JOURNAL.jsonl'
    if journal.exists():
        lines=journal.read_text().splitlines()
        for i,line in enumerate(lines):
            try:record=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                break
            key=tuple(record['key'])
            if key in completed:raise ValueError('duplicate label receipt')
            completed[key]=record
        journal.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in completed.values()))
    for split in ['train_pbr','val_pbr']:
        renderer=make_renderer(data,split,obj);renderer.worker_init_fn(0)
        entries=renderer.dataset_info['obj_info'][f'obj_{obj:06d}']
        for i,info in enumerate(entries):
            paths=[root/(split+'_xyz_GT_'+s)/info['scene']/(Path(info['mask_path']).stem+'.png') for s in ['front','back']]
            key=(split,info['scene'],Path(info['mask_path']).stem)
            if key in completed:
                for r in completed[key]['maps']:
                    if sha(root/r['file'])!=r['sha256']:raise ValueError('previously verified label changed')
            else:
                renderer[i]
                front=renderer.renderer_vispy.front_depth;back=np.asarray(renderer.renderer_vispy.depth)
                if not (front>0).any() or not np.isfinite(front).all() or not np.isfinite(back).all():raise ValueError('invalid rendered coordinate support')
                if np.mean(back[front>0]>=front[front>0]-.1)<.99:raise ValueError('front/back depth order differs')
                visible=cv2.imread(info['mask_visib_path'],0)
                if visible is None or visible.shape!=front.shape:raise ValueError('invalid original visible mask')
                expanded=cv2.dilate((front>0).astype(np.uint8),np.ones((3,3),np.uint8))
                # Valid empty masks remain valid original supervision, not missing data.
                outside=float(np.mean(expanded[visible>0]==0)) if visible.any() else 0.
                if outside>.05:raise ValueError('CAD/native mask alignment failed')
            if i==0:
                vendor=GL.glGetString(GL.GL_VENDOR).decode();device=GL.glGetString(GL.GL_RENDERER).decode()
                if 'NVIDIA' not in vendor or 'llvmpipe' in device.lower():raise ValueError('hardware EGL required')
            for path in paths:
                image=cv2.imread(str(path))
                if image is None or image.shape!=(info['height'],info['width'],3) or not image.any():raise ValueError('invalid dense coordinate map')
                records.append({'file':str(path.relative_to(root)),'sha256':sha(path),'bytes':path.stat().st_size})
            if key not in completed:
                receipt={'key':list(key),'maps':records[-2:]}
                with journal.open('a') as stream:stream.write(json.dumps(receipt,allow_nan=False)+'\n');stream.flush()
                completed[key]=receipt
            if (i+1)%100==0:
                save(root/'LABEL_PROGRESS.json',{'split':split,'done':i+1,'total':len(entries),'time':time.time()})
                print('HCCE_LABEL_PROGRESS',ready['dataset'],obj,split,i+1,len(entries),flush=True)
    save(root/'LABELS_READY.json',{'identity':identity,'maps':records,'map_count':len(records),'bytes':sum(r['bytes'] for r in records)})
    print('HCCE_LABELS_COMPLETE',ready['dataset'],obj,len(records),flush=True)


def loss_state(criterion):
    return {side:{'history':[[copy.deepcopy(bit.data) for bit in axis] for axis in getattr(criterion,side+'_error_list')],
                  'weights':[None if v is None else v.detach().cpu() for v in getattr(criterion,'weight_'+side.lower()+'_error_ratio')]}
            for side in ['Front','Back']}


def restore_loss(criterion,state,device):
    from HccePose.network_model import FixedSizeList
    for side in ['Front','Back']:
        history=[]
        for axis in state[side]['history']:
            lists=[]
            for values in axis:
                item=FixedSizeList(100)
                for value in values:item.append(value)
                lists.append(item)
            history.append(lists)
        setattr(criterion,side+'_error_list',history)
        setattr(criterion,'weight_'+side.lower()+'_error_ratio',[None if v is None else v.to(device) for v in state[side]['weights']])


def safe_pnp_backend():
    from HccePose import PnP_solver
    if getattr(PnP_solver.solve_PnP,'hcce_failure_guard',False):return PnP_solver
    original=PnP_solver.solve_PnP
    def safe(*args,**kwargs):
        try:
            result=original(*args,**kwargs)
            if result['success'] and np.isfinite(result['rot']).all() and np.isfinite(result['tvecs']).all():return result
        except cv2.error:
            pass
        return {'success':False,'rot':np.eye(3),'tvecs':np.zeros((3,1)),'inliers':np.empty((0,1),dtype=int)}
    safe.hcce_failure_guard=True;PnP_solver.solve_PnP=safe;return PnP_solver


def validate(model,loader,mesh,info):
    from HccePose.metric import add_s
    solver=safe_pnp_backend();combos=[tuple(c) for count in range(4,0,-1) for c in itertools.combinations(range(4),count)]
    values=[];valid_candidates=0;model.eval()
    with torch.inference_mode():
        for batch in loader:
            rgb,mask,front,back,boxes,K,R,t=batch
            result=model.inference_batch(rgb.cuda(),boxes.cuda())
            masks=result['pred_mask'].cpu().numpy();a=result['pred_front_code_obj'].cpu().numpy();b=result['pred_back_code_obj'].cpu().numpy();uv=result['coord_2d_image'].cpu().numpy()
            if not all(np.isfinite(x).all() for x in [masks,a,b,uv]):raise ValueError('nonfinite validation outputs')
            for i in range(len(rgb)):
                candidates=solver.solve_PnP_comb((masks[i],a[i],b[i],uv[i],K[i].numpy()),train=True)
                accuracy=[]
                for c in candidates:
                    valid_candidates+=int(c['success'])
                    accuracy.append(float(add_s(mesh,info,[[R[i].numpy(),t[i].numpy()]],[[c['rot'],c['tvecs']]])[0]) if c['success'] else 0.)
                row=[]
                for combo in combos:
                    best=max(combo,key=lambda j:candidates[j]['num'])
                    row.append(accuracy[best] if candidates[best]['num']>0 else 0.)
                values.append(row)
    model.train()
    if not values:raise ValueError('empty held-out synthetic validation')
    scores=np.mean(values,axis=0)
    return scores,{'samples':len(values),'candidate_successes':valid_candidates,'ADD_ADDS_success_best_combo':float(scores.max()),'combo_scores':scores.tolist(),
                   'population':'deterministic held-out synthetic base-val subset, not training samples or real/test selection'}


def train(root,iterations,resume=False,workers=8):
    from HccePose.network_model import HccePose_BF_Net,HccePose_Loss
    from bop_toolkit.bop_toolkit_lib.inout import load_ply
    ready=verify_inputs(root);obj=ready['object'];data=FrozenBOP(root)
    contract=json.loads((root.parents[3]/'contract'/ready['dataset']/'PASSED.json').read_text())
    if not contract['passed'] or contract['train_driver_sha256']!=sha(__file__):raise ValueError('production API contract is missing or stale')
    labels_ready=json.loads((root/'LABELS_READY.json').read_text())
    if labels_ready['identity']!=label_identity(root):raise ValueError('label identity differs')
    for rec in labels_ready['maps']:
        if sha(root/rec['file'])!=rec['sha256']:raise ValueError('coordinate map changed')
    if iterations!=50000:raise ValueError('production fixed budget is50000 iterations; use separate smoke entrypoint')
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    b=backend();dataset=b.train_bop_dataset_back_front(data,'train_pbr',padding_ratio=1.5,crop_size_img=256)
    dataset.update_obj_id(obj,data.obj_model_list[0]);valdata=b.test_bop_dataset_back_front(data,'val_pbr',padding_ratio=1.5,crop_size_img=256,ratio=1.)
    valdata.update_obj_id(obj,data.obj_model_list[0]);count=min(len(valdata),max(1,len(dataset)//100));indices=np.linspace(0,len(valdata)-1,count).astype(int).tolist()
    loader=torch.utils.data.DataLoader(dataset,batch_size=24,shuffle=True,num_workers=workers,drop_last=True)
    if not len(loader):raise ValueError('empty complete training batches')
    val_loader=torch.utils.data.DataLoader(torch.utils.data.Subset(valdata,indices),batch_size=24,shuffle=False,num_workers=min(workers,4))
    info=data.model_info[str(obj)];minimum=torch.tensor([info['min_'+s] for s in 'xyz'],dtype=torch.float32,device='cuda');size=torch.tensor([info['size_'+s] for s in 'xyz'],dtype=torch.float32,device='cuda')
    model=HccePose_BF_Net(efficientnet_key=None,input_channels=3,min_xyz=minimum,size_xyz=size).cuda().train()
    criterion=HccePose_Loss();optimizer=torch.optim.Adam(model.parameters(),lr=.0002);scaler=torch.cuda.amp.GradScaler()
    work=root.parent/'work';work.mkdir(exist_ok=True);identity={'inputs_sha256':sha(root/'INPUTS_READY.json'),'labels_sha256':sha(root/'LABELS_READY.json'),
        'driver_sha256':sha(__file__),'adapter_sha256':sha(Path(__file__).with_name('hccepose_data.py')),'model':'upstream HCCEPose BF ResNet34 ImageNet',
        'iterations':50000,'batch':24,'lr':.0002,'seed':42,'crop':256,'padding':1.5,'validation_indices':indices,
        'no_DOPER_weight_initialization':True,'fixed_final_endpoint':True,'real_validation_used_for_selection':False}
    ip=work/'IDENTITY.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('immutable training identity changed')
    if (work/'last.pth').exists() and not resume:raise FileExistsError('checkpoint exists; explicit same-recipe resume required')
    save(ip,identity);step=0;updates=0;skipped=0;prior=np.zeros(15);mesh=load_ply(data.obj_model_list[0]);last_val=None
    if resume:
        ck=torch.load(work/'last.pth',map_location='cpu',weights_only=False)
        if ck['experiment_identity']!=identity:raise ValueError('resume identity mismatch')
        model.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict']);scaler.load_state_dict(ck['scaler_state_dict']);restore_loss(criterion,ck['criterion_history'],'cuda')
        step=ck['iteration_step'];updates=ck['successful_updates'];skipped=ck['skipped_updates'];prior=np.asarray(ck['keypoints_'])
        torch.set_rng_state(ck['rng']['torch']);np.random.set_state(ck['rng']['numpy']);random.setstate(ck['rng']['python']);torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    start=time.perf_counter()
    while step<iterations:
        for rgb,mask,front,back in loader:
            rgb,mask,front,back=[x.cuda(non_blocking=True) for x in [rgb,mask,front,back]];optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred_mask,codes=model(rgb);result=criterion(codes[:,:24],codes[:,24:],pred_mask,front,back,mask)
                loss=3*result['Front_L1Losses'].sum()+3*result['Back_L1Losses'].sum()+result['mask_loss']
            if not torch.isfinite(loss):raise ValueError('nonfinite loss; no silent BN reset or batch skipping')
            scaler.scale(loss).backward();old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            did_update=scaler.get_scale()>=old;updates+=int(did_update);skipped+=int(not did_update);step+=1
            if step%20==0:print('HCCE_TRAIN',ready['dataset'],obj,step,iterations,'loss',float(loss.detach()),'updates',updates,flush=True)
            if step%500==0 or step==iterations:
                prior,last_val=validate(model,val_loader,mesh,info)
                if any(v.is_floating_point() and not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite model weights')
                checkpoint={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'iteration_step':step,
                    'best_score':float(prior.max()),'keypoints_':prior.tolist(),'scaler_state_dict':scaler.state_dict(),'criterion_history':loss_state(criterion),
                    'successful_updates':updates,'skipped_updates':skipped,'experiment_identity':identity,'validation':last_val,
                    'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}}
                atomic_checkpoint(work/'last.pth',checkpoint)
                save(work/'PROGRESS.json',{'iteration':step,'target':iterations,'successful_updates':updates,'skipped_updates':skipped,'loss':float(loss.detach()),'validation':last_val,'elapsed_seconds':time.perf_counter()-start,'time':time.time()})
                print('HCCE_CHECKPOINT',obj,step,'synthetic_val_ADD',float(prior.max()),flush=True)
            if step>=iterations:break
    save(work/'COMPLETE.json',{'completed':True,'iterations':step,'checkpoint_sha256':sha(work/'last.pth'),'successful_updates':updates,'AMP_skips':skipped,'selected_endpoint':'fixed final50000, not real-val/test best','synthetic_validation':last_val})
    print('HCCE_TRAIN_COMPLETE',ready['dataset'],obj,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['labels','train']);p.add_argument('--root',type=Path,required=True);p.add_argument('--iterations',type=int,default=50000);p.add_argument('--workers',type=int,default=8);p.add_argument('--resume',action='store_true');p.add_argument('--allow-nonstandard-backbone',action='store_true');a=p.parse_args()
    if not a.allow_nonstandard_backbone:p.error('explicit upstream ResNet reproduction opt-out required')
    a.root.parent.mkdir(parents=True,exist_ok=True)
    with (a.root.parent/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.mode=='labels':labels(a.root)
        else:train(a.root,a.iterations,a.resume,a.workers)
