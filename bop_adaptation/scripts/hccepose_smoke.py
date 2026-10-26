#!/usr/bin/env python3
"""GPU geometry/label/optimizer smoke for upstream HCCEPose, never a final model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from hccepose_paths import UPSTREAM
from hccepose_data import FrozenBOP,backend,make_renderer,sha,save


def atomic_torch_save(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(path)


def render_labels(root,obj):
    from OpenGL import GL
    data=FrozenBOP(root);audit={};previews=root/'review';previews.mkdir(exist_ok=True)
    b=backend()
    for split in ['train_pbr','val_pbr']:
        renderer=make_renderer(data,split,obj);renderer.worker_init_fn(0)
        entries=renderer.dataset_info['obj_info'][f'obj_{obj:06d}'];ratios=[];errors=[];sizes=[]
        for i,info in enumerate(entries):
            renderer[i]
            front=renderer.renderer_vispy.front_depth;back=np.asarray(renderer.renderer_vispy.depth)
            if not (front>0).any() or not np.isfinite(front).all() or not np.isfinite(back).all():raise ValueError('invalid depth labels')
            if np.mean(back[front>0]>=front[front>0]-.1)<.99:raise ValueError('front/back depth ordering differs')
            names=[root/(split+'_xyz_GT_'+side)/info['scene']/(Path(info['mask_path']).stem+'.png') for side in ['front','back']]
            maps=[cv2.imread(str(p)) for p in names]
            if any(m is None or m.shape[:2]!=front.shape for m in maps):raise ValueError('coordinate image missing or wrong shape')
            mask=cv2.imread(info['mask_visib_path'],0)
            if mask is None or mask.shape!=front.shape or not (mask>0).any():raise ValueError('original visible mask invalid')
            expanded=cv2.dilate((front>0).astype(np.uint8),np.ones((3,3),np.uint8))
            outside=float(np.mean(expanded[mask>0]==0));ratios.append(outside)
            if outside>.05:raise ValueError('original mask does not align with CAD depth silhouette')
            coords=np.argwhere(front>0);coords=coords[::max(1,len(coords)//500)]
            R,t=renderer.last_canonical_rt;K=np.asarray(info['cam_K']).reshape(3,3)
            for m in maps:
                points=m[coords[:,0],coords[:,1]].astype(float)/255*renderer.div_v+renderer.min_xyz
                camera=points@R.T+np.asarray(t).reshape(3)
                projected=camera@K.T;uv=projected[:,:2]/projected[:,2:]
                err=float(np.median(np.linalg.norm(uv-coords[:,::-1],axis=1)));errors.append(err)
                if not np.isfinite(err) or err>3:raise ValueError('dense-coordinate roundtrip exceeds quantization tolerance')
            sizes.extend(p.stat().st_size for p in names)
            if i in [0,len(entries)//2,len(entries)-1]:
                box=b.pad_square_fp32(np.asarray(info['bbox_visib']),1.5)
                rgb=cv2.imread(info['rgb']);mask_show=rgb.copy()
                for shape,color in [(mask>0,(0,255,255)),(front>0,(255,255,0))]:
                    cv2.drawContours(mask_show,cv2.findContours(shape.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,color,2)
                panels=[b.crop_square_resize(x,box,384,cv2.INTER_NEAREST if k else cv2.INTER_LINEAR) for k,x in enumerate([mask_show,*maps])]
                for panel,label in zip(panels,['RGB: visible yellow / CAD cyan','Front XYZ coordinate target','Back XYZ coordinate target']):
                    cv2.putText(panel,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                cv2.imwrite(str(previews/f'{split}_{i:06d}.jpg'),np.concatenate(panels,axis=1))
        gl={k:GL.glGetString(v).decode(errors='replace') for k,v in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]}
        if 'NVIDIA' not in gl['vendor'] or 'llvmpipe' in gl['renderer'].lower():raise ValueError('hardware NVIDIA EGL renderer required')
        audit[split]={'samples':len(entries),'max_visible_mask_outside_tolerance_fraction':max(ratios),
                      'max_median_coordinate_roundtrip_px':max(errors),'mean_label_bytes':float(np.mean(sizes)),'GL':gl}
        print('DENSE_LABEL_SMOKE_OK',split,audit[split],flush=True)
    save(root/'LABEL_SMOKE.json',audit);return audit


def run(root,iterations):
    from HccePose.network_model import HccePose_BF_Net,HccePose_Loss
    source=json.loads((root/'INPUTS_READY.json').read_text())
    if source['smoke_only'] is not True or not 1<=iterations<=100:raise ValueError('bounded separate smoke dataset required')
    if not torch.cuda.is_available():raise RuntimeError('cluster GPU required')
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
    obj=source['object'];data=FrozenBOP(root)
    identity={'upstream_revision':'f75cc4d5db2fc298fc7b853f8c75273757872382','inputs_sha256':sha(root/'INPUTS_READY.json'),
              'driver_sha256':sha(__file__),'adapter_sha256':sha(Path(__file__).with_name('hccepose_data.py')),
              'upstream_files':{name:sha(UPSTREAM/name) for name in ['HccePose/bop_loader.py','HccePose/network_model.py','HccePose/PnP_solver.py']},
              'model_family':'HCCEPose_BF_upstream_ResNet34','allow_nonstandard_backbone':True,'scope':'geometry and finite training smoke only',
              'cuda':torch.cuda.get_device_name(0),'iterations':iterations,'batch':24}
    if (root/'SMOKE_IDENTITY.json').exists() and json.loads((root/'SMOKE_IDENTITY.json').read_text())!=identity:raise ValueError('smoke identity differs; new root required')
    save(root/'SMOKE_IDENTITY.json',identity)
    labels=render_labels(root,obj)
    b=backend();train=b.train_bop_dataset_back_front(data,'train_pbr',padding_ratio=1.5,crop_size_img=256)
    train.update_obj_id(obj,data.obj_model_list[0])
    loader=torch.utils.data.DataLoader(train,batch_size=24,shuffle=True,num_workers=0,drop_last=True)
    info=data.model_info[str(obj)];minimum=torch.tensor([info['min_'+s] for s in 'xyz'],dtype=torch.float32,device='cuda');size=torch.tensor([info['size_'+s] for s in 'xyz'],dtype=torch.float32,device='cuda')
    model=HccePose_BF_Net(efficientnet_key=None,input_channels=3,min_xyz=minimum,size_xyz=size).cuda().train()
    criterion=HccePose_Loss();optimizer=torch.optim.Adam(model.parameters(),lr=.0002);scaler=torch.cuda.amp.GradScaler()
    work=root/'smoke_training';work.mkdir(exist_ok=True)
    if any(work.iterdir()):raise FileExistsError('smoke output exists; inspect rather than overwrite')
    updates=0;losses=[];start=time.perf_counter();step=0
    while step<iterations:
        for rgb,mask,front,back in loader:
            rgb,mask,front,back=[x.cuda() for x in (rgb,mask,front,back)]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred_mask,codes=model(rgb)
                result=criterion(codes[:,:24],codes[:,24:],pred_mask,front,back,mask)
                loss=3*torch.sum(result['Front_L1Losses'])+3*torch.sum(result['Back_L1Losses'])+result['mask_loss']
            if not torch.isfinite(loss):raise ValueError('nonfinite upstream training loss')
            scaler.scale(loss).backward();oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            updates+=int(scaler.get_scale()>=oldscale);losses.append(float(loss));step+=1
            print('HCCE_SMOKE_STEP',source['dataset'],obj,step,'loss',losses[-1],'optimizer_updates',updates,flush=True)
            if step>=iterations:break
    if updates<1 or any(v.is_floating_point() and not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('smoke did not make finite model updates')
    atomic_torch_save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'iteration_step':step,
                       'best_score':0.,'keypoints_':[0.]*15,'scaler_state_dict':scaler.state_dict(),'experiment_identity':identity},work/'last.pth')
    validation=b.test_bop_dataset_back_front(data,'val_pbr',padding_ratio=1.5,crop_size_img=256,ratio=1.0)
    validation.update_obj_id(obj,data.obj_model_list[0]);vloader=torch.utils.data.DataLoader(validation,batch_size=4,shuffle=False,num_workers=0)
    model.eval();val=[]
    with torch.no_grad():
        for batch in vloader:
            rgb,mask,front,back,boxes,K,R,t=batch
            result=model.inference_batch(rgb.cuda(),boxes.cuda())
            if any(not torch.isfinite(result[k]).all() for k in ['pred_mask','pred_front_code_obj','pred_back_code_obj']):raise ValueError('nonfinite validation outputs')
            val.append(int(rgb.shape[0]))
    save(root/'SMOKE_PASSED.json',{'passed':True,'iterations':step,'successful_updates':updates,'losses':losses,
        'synthetic_val_forward_samples':sum(val),'coordinate_label_checks':labels,'elapsed_seconds':time.perf_counter()-start,
        'not_full_training':True,'not_real_validation_accuracy':True,'checkpoint_sha256':sha(work/'last.pth')})
    print('HCCE_SMOKE_PASSED',source['dataset'],obj,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--iterations',type=int,default=20);p.add_argument('--allow-nonstandard-backbone',action='store_true');a=p.parse_args()
    if not a.allow_nonstandard_backbone:p.error('explicit upstream ResNet34 reproduction opt-out required')
    policy=json.loads((a.root/'INPUTS_READY.json').read_text()).get('render_compatibility')
    if policy:
        from hccepose_label_clipping import install
        install(policy)
    run(a.root,a.iterations)
