#!/usr/bin/env python3
"""Conditional positive-near-plane fix for newly declared HCCE label recipes.

Regular positive-near-plane renders delegate to the unchanged upstream method.
Only invalid nonpositive planes use the same view/projection/draw/depth code with
an explicit small positive near plane. Original CAD and R/t/K are unchanged.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
from hccepose_data import backend,sha,save

from hccepose_paths import UPSTREAM as CHECKOUT
UPSTREAM=CHECKOUT/'bop_toolkit/bop_toolkit_lib/renderer_vispy.py'


def recipe():
    return {'policy':'positive near plane only when original bbox near<=0; unchanged positive branch',
            'extension_sha256':sha(__file__),'upstream_renderer_sha256':sha(UPSTREAM),
            'near_cap_mm':1.,'near_far_fraction':.001}


def clipping_planes(depths):
    z=np.asarray(depths,float)
    if not np.isfinite(z).all() or z.size==0:raise ValueError('invalid camera bounding-box depths')
    near=float(z.min());far=float(z.max())
    if far<=0:raise ValueError('whole object is behind camera')
    if far<=near:raise ValueError('degenerate camera depth interval')
    if near>0:return near,far,False
    return min(1.,far*.001),far,True


def install(policy):
    if policy!=recipe():raise ValueError('renderer compatibility recipe changed')
    module=backend().renderer;original_factory=module.create_renderer
    if hasattr(original_factory,'hcce_clipping_stats'):return original_factory.hcce_clipping_stats
    stats={'policy':policy,'renders':0,'corrected_renders':0,'sampled_corrections':[]}
    def factory(*args,**kwargs):
        renderer=original_factory(*args,**kwargs)
        implementation=sys.modules[type(renderer).__module__]
        if sha(implementation.__file__)!=policy['upstream_renderer_sha256']:raise ValueError('unexpected upstream renderer')
        original=renderer.render_object
        def render(obj_id,R,t,fx,fy,cx,cy,clear=True,draw_back=False):
            view_cv=np.eye(4,dtype=np.float32);view_cv[:3,:3]=R;view_cv[:3,3]=np.asarray(t).squeeze()
            corners=renderer.model_bbox_corners[obj_id]
            homogeneous=np.concatenate((corners,np.ones((len(corners),1))),axis=1).T
            depths=view_cv[2,:].reshape(1,4).dot(homogeneous)
            near,far,corrected=clipping_planes(depths);stats['renders']+=1
            if not corrected:return original(obj_id,R,t,fx,fy,cx,cy,clear=clear,draw_back=draw_back)
            stats['corrected_renders']+=1
            if len(stats['sampled_corrections'])<32:
                stats['sampled_corrections'].append({'old_near_mm':float(depths.min()),'near_mm':near,'far_mm':far,'draw_back':bool(draw_back)})
            renderer.clip_near=near;renderer.clip_far=far
            model=np.eye(4,dtype=np.float32);view=renderer.pose_cv_to_gl.dot(view_cv).T
            K=np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
            projection=implementation._calc_calib_proj(K,0,0,renderer.width,renderer.height,near,far)
            renderer.update();renderer.on_draw(obj_id,model,view,projection,clear=clear,draw_back=draw_back)
            if renderer.mode=='depth':return {'depth':renderer.depth}
            if renderer.mode=='rgb':return {'rgb':renderer.rgb}
            if renderer.mode=='rgb+depth':return {'rgb':renderer.rgb,'depth':renderer.depth}
            raise ValueError('unsupported renderer mode')
        renderer.render_object=render
        return renderer
    factory.hcce_clipping_stats=stats;module.create_renderer=factory
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--allow-nonstandard-backbone',action='store_true');a=p.parse_args()
    if not a.allow_nonstandard_backbone:p.error('explicit HCCEPose reproduction opt-out required')
    source=json.loads((a.root/'INPUTS_READY.json').read_text())
    policy=source.get('render_compatibility')
    if policy is None:raise ValueError('input does not declare the conditional clipping recipe')
    with (a.root.parent/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from hccepose_train import labels
        stats=install(policy);labels(a.root)
        save(a.root/'LABEL_RENDER_COMPAT.json',stats)
        ready=json.loads((a.root/'LABELS_READY.json').read_text())
        ready['renderer_extension']={'receipt_sha256':sha(a.root/'LABEL_RENDER_COMPAT.json'),**policy}
        save(a.root/'LABELS_READY.json',ready)
        print('HCCE_CLIPPING_LABELS_COMPLETE',stats['corrected_renders'],flush=True)
