"""Replay frozen DOPER base COCO samples through the upstream HCCEPose loaders.

Original RGB, masks, object frames and splits are never rewritten. New front/back
coordinate maps live in a separate experiment. No zero-label or guessed-K fallback.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from hccepose_paths import UPSTREAM
# hccepose_paths installs both package-qualified and legacy BOP import roots.
CV_TO_GL = np.diag([1.,-1.,-1.,1.])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def annotation_pose(ann):
    """Frozen COCO GL-camera/world metre transforms -> original BOP CV pose mm."""
    camera=np.asarray(ann['camera_extrinsics'],float);obj=np.asarray(ann['object_pose'],float)
    if camera.shape!=(4,4) or obj.shape!=(4,4) or not np.isfinite(camera).all() or not np.isfinite(obj).all():
        raise ValueError('finite original pose transforms required')
    T=CV_TO_GL@np.linalg.inv(camera)@obj
    if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-4):
        raise ValueError('invalid rigid pose')
    return T[:3,:3],T[:3,3:4]*1000


def intrinsics(ann):
    c=ann['camera_intrinsics'];K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]],float)
    if not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError('native per-frame K required')
    return K


def reprojection_error(ann,points_m):
    R,t=annotation_pose(ann);xyz=np.asarray(points_m)@R.T+t.reshape(3)/1000
    if np.any(xyz[:,2]<=0):raise ValueError('landmark behind camera')
    q=xyz@intrinsics(ann).T;uv=q[:,:2]/q[:,2:]
    stored=np.asarray(ann['keypoints'],float).reshape(-1,3)
    valid=stored[:,2]>0
    if len(stored)!=len(points_m) or not valid.any():raise ValueError('invalid frozen keypoint record')
    return float(np.max(np.linalg.norm(uv[valid]-stored[valid,:2],axis=1)))


def symmetry_matrices(items):
    """Bundled BOP returns {R,t} records for continuous symmetries; normalize them."""
    result=[]
    for s in items:
        if isinstance(s,dict):
            T=np.eye(4);T[:3,:3]=np.asarray(s['R']).reshape(3,3);T[:3,3]=np.asarray(s['t']).reshape(3)
        else:T=np.asarray(s,float).reshape(4,4)
        if not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('invalid symmetry transform')
        result.append(T.ravel().tolist())
    return result


class StrictCV2:
    """Same OpenCV operations; reject unreadable data instead of silent zero maps."""
    def __init__(self,cv2):self.backend=cv2
    def __getattr__(self,name):return getattr(self.backend,name)
    def imread(self,path,*args,**kwargs):
        image=self.backend.imread(path,*args,**kwargs)
        if image is None:raise FileNotFoundError('required HCCE RGB/mask/coordinate map unreadable: '+str(path))
        return image
    def imwrite(self,path,*args,**kwargs):
        result=self.backend.imwrite(path,*args,**kwargs)
        if not result:raise OSError('HCCE coordinate map write failed: '+str(path))
        return result


class OriginalPoseDepthRenderer:
    """Render BOTH surfaces at the original GT pose; canonicalization only changes XYZ labels.

Upstream overwrites local R before the back render but keeps original t, which
is inconsistent for offset symmetry transforms. No CAD/GT recentering is needed.
"""
    def __init__(self,renderer):self.renderer=renderer;self.original=None
    def __getattr__(self,name):return getattr(self.renderer,name)
    def render_object(self,obj_id,R,t,fx,fy,cx,cy,draw_back=False,**kwargs):
        if not draw_back:self.original=(np.asarray(R).copy(),np.asarray(t).copy())
        elif self.original is None:raise ValueError('back render requires preceding front render')
        if draw_back:R,t=self.original
        result=self.renderer.render_object(obj_id,R,t,fx,fy,cx,cy,draw_back=draw_back,**kwargs)
        if not draw_back and hasattr(self.renderer,'depth'):self.front_depth=np.asarray(self.renderer.depth).copy()
        return result


def backend():
    from HccePose import bop_loader
    if not isinstance(bop_loader.cv2,StrictCV2):bop_loader.cv2=StrictCV2(bop_loader.cv2)
    return bop_loader


class FrozenBOP:
    """Duck-typed bop_dataset retaining exactly the archived per-object crop list."""
    def __init__(self,root):
        self.root=Path(root);self.dataset_path=str(self.root);self.dataset_name=self.root.name;self.local_rank=0
        self.model_path=str(self.root/'models');self.model_info=json.loads((self.root/'models/models_info.json').read_text())
        self.obj_id_list=list(map(int,self.model_info));self.obj_model_list=[str(self.root/f'models/obj_{o:06d}.ply') for o in self.obj_id_list]
        self.obj_info_list=[self.model_info[str(o)] for o in self.obj_id_list]
    def load_folder(self,folder_name,scene_num=200,vis=0.0):
        if folder_name not in ['train_pbr','val_pbr']:raise ValueError('only frozen synthetic splits allowed')
        rows=json.loads((self.root/(folder_name+'.json')).read_text())
        obj=self.obj_id_list[0]
        print('FROZEN_COCO_SELECTION',folder_name,len(rows),'upstream_extra_visibility_filter_not_applied',vis,flush=True)
        return {'obj_info':{f'obj_{obj:06d}':rows},'img_info':{},
                'scene_path_list':[str(self.root/folder_name/s) for s in sorted({r['scene'] for r in rows})]}


def make_renderer(data,split,obj):
    b=backend()
    class Renderer(b.rendering_bop_dataset_back_front):
        def __getitem__(self,index):
            info=self.dataset_info['obj_info'][f'obj_{self.current_obj_id:06d}'][index]
            self.last_canonical_rt=(np.asarray(info['cam_R_m2c']).reshape(3,3),np.asarray(info['cam_t_m2c']).reshape(3,1))
            seed=int(hashlib.sha256((self.folder_name+str(self.current_obj_id)+info['scene']+info['image']).encode()).hexdigest()[:8],16)%2147483647
            np.random.seed(seed);b.cv2.setRNGSeed(seed)
            return super().__getitem__(index)
        def modified_sym_Rt(self,*args,**kwargs):
            result=super().modified_sym_Rt(*args,**kwargs);self.last_canonical_rt=result;return result
        def update_obj_id(self,oid,path):
            super().update_obj_id(oid,path)
            if 'symmetries_discrete' in self.model_info_obj:
                self.model_info_obj['symmetries_discrete']=symmetry_matrices(self.model_info_obj['symmetries_discrete'])
        def worker_init_fn(self,worker_id):
            self.worker_id=worker_id
            first=self.dataset_info['obj_info'][f'obj_{self.current_obj_id:06d}'][0]
            self.img_shape=b.cv2.imread(first['rgb']).shape[:2]
            renderer=b.renderer.create_renderer(self.img_shape[1],self.img_shape[0],'vispy',mode='depth',shading='flat')
            # A depth target has no material/texture dependence; avoid optional missing texture assets.
            renderer.add_object(0,self.current_obj_path,surf_color=[1.,1.,1.])
            self.renderer_vispy=OriginalPoseDepthRenderer(renderer)
            vertices=b.inout.load_ply(self.current_obj_path)['pts'];self.min_xyz=np.min(vertices,axis=0);self.div_v=np.ptp(vertices,axis=0)
            if np.any(self.div_v<=1e-8):raise ValueError('degenerate object extent')
    r=Renderer(data,split);r.update_obj_id(obj,data.obj_model_list[0]);return r


def prepare_object(source_record,inventory,out,smoke=False):
    import trimesh
    source=Path(source_record['base_root']);obj=source_record['object'];out=Path(out)
    if out.exists():raise FileExistsError('new HCCE preparation root required')
    keypoints=Path(source_record['keypoints'])
    if sha(keypoints)!=source_record['keypoints_sha256']:raise ValueError('original geometry identity changed')
    points=np.asarray(json.loads(keypoints.read_text())['keypoints_3d'],float)
    mesh_path=source/f'geometry/obj_{obj:06d}.ply';mesh=trimesh.load_mesh(mesh_path,process=False,skip_materials=True)
    vertices=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces)
    if not np.isfinite(vertices).all() or not len(faces) or faces.shape[1]!=3:raise ValueError('original finite triangular CAD required')
    minimum=vertices.min(0);maximum=vertices.max(0);size=maximum-minimum
    if np.any(size<=1e-8):raise ValueError('nondegenerate original mesh required')
    if np.any(points*1000<minimum-1) or np.any(points*1000>maximum+1):raise ValueError('mesh/landmark mm/m frame mismatch')
    info=json.loads((source/'geometry/models_info.json').read_text())[str(obj)];original_info=copy.deepcopy(info)
    for i,axis in enumerate('xyz'):info.update({f'min_{axis}':float(minimum[i]),f'max_{axis}':float(maximum[i]),f'size_{axis}':float(size[i])})
    gathered={};audits={}
    for split,name,limit in [('train','train_pbr',64),('val','val_pbr',16)]:
        path=source/f'dataset/{split}/annotations.json'
        if sha(path)!=inventory['splits'][split]['sha256']:raise ValueError('frozen base COCO changed')
        coco=json.loads(path.read_text());images={i['id']:i for i in coco['images']}
        if (len(coco['annotations'])!=len(images)
                or {a['image_id'] for a in coco['annotations']}!=set(images)):
            raise ValueError('expected exact historical one-crop-per-image base selection')
        anns=sorted(coco['annotations'],key=lambda a:(a['image_id'],a['id']))
        if smoke:anns=[anns[i] for i in sorted(set(np.linspace(0,len(anns)-1,min(limit,len(anns))).astype(int)))]
        rows=[];names=set();errors=[]
        for ann in anns:
            image=images[ann['image_id']];rgb=source/f'dataset/{split}/images'/image['file_name'];mask=source/f'dataset/{split}'/ann['mask_file']
            if not rgb.is_file() or not mask.is_file():raise ValueError('missing original RGB/mask')
            R,t=annotation_pose(ann);K=intrinsics(ann);err=reprojection_error(ann,points)
            if err>.1:raise ValueError('COCO -> BOP frame/calibration reprojection mismatch')
            errors.append(err);scene=str(ann.get('bop',{}).get('scene','000000'));stem=mask.stem
            if (scene,stem) in names:raise ValueError('label filename collision')
            names.add((scene,stem))
            rows.append({'obj_id':obj,'scene':scene,'image':str(ann.get('bop',{}).get('frame',image['id'])).zfill(6),
                'rgb':str(rgb.resolve()),'mask_visib_path':str(mask.resolve()),
                'mask_path':str(mask),'mask_path_role':'label-filename identifier only; actual supervision is original visible mask',
                'cam_K':K.ravel().tolist(),'cam_R_m2c':R.ravel().tolist(),'cam_t_m2c':t.ravel().tolist(),
                'bbox_visib':ann['bbox'],'bbox_source':'exact retained COCO box; no new crop selection',
                'width':image['width'],'height':image['height'],'coco_image_id':image['id'],'coco_annotation_id':ann['id']})
        gathered[name]=rows;audits[split]={'source_sha256':sha(path),'samples':len(rows),'max_landmark_reprojection_px':max(errors)}
    train={r['rgb'] for r in gathered['train_pbr']};val={r['rgb'] for r in gathered['val_pbr']}
    if train&val:raise ValueError('frozen train/val overlap')
    out.mkdir(parents=True);(out/'models').mkdir()
    shutil.copyfile(mesh_path,out/f'models/obj_{obj:06d}.ply')
    save(out/'models/models_info.json',{str(obj):info});save(out/'ORIGINAL_MODELS_INFO.json',{str(obj):original_info})
    for name,rows in gathered.items():save(out/(name+'.json'),rows)
    proof={'dataset':source_record['dataset'],'object':obj,'source_base_root':str(source),'source_checkpoint_used_for_training':False,
        'source_keypoints_sha256':sha(keypoints),'mesh_sha256':sha(mesh_path),'source_split_audits':audits,
        'RGB_and_masks_reused_unchanged':True,'object_frame_changed':False,'mesh_pose_units':'mm',
        'native_K_per_sample':True,'sampling':'exact frozen COCO retained instance per image; no extra upstream visibility cutoff',
        'smoke_only':smoke,'model_info_sha256':sha(out/'models/models_info.json'),
        'manifests_sha256':{name:sha(out/(name+'.json')) for name in gathered}}
    save(out/'INPUTS_READY.json',proof);return proof
