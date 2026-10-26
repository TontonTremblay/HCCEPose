"""Preparation-only COCO visibility correction; frozen loaders/render/train stay intact.

COCO v=0 points have no 2-D supervision and may be behind the camera. Every
labelled point (v=1 or2) still requires positive depth and the original0.1px
reprojection gate. Never filter rows or modify their poses/masks/crop boxes.
"""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import hccepose_data as legacy


def visibility_reprojection(ann,points_m):
    points=np.asarray(points_m,dtype=float);stored=np.asarray(ann['keypoints'],dtype=float).reshape(-1,3)
    if points.ndim!=2 or points.shape[1]!=3 or stored.shape!=points.shape:
        raise ValueError('invalid frozen keypoint shape')
    if not np.isfinite(points).all() or not np.isfinite(stored).all() or not np.isin(stored[:,2],[0,1,2]).all():
        raise ValueError('invalid frozen keypoint values/visibility')
    labelled=stored[:,2]>0
    if not labelled.any():raise ValueError('no labelled keypoints')
    R,t=legacy.annotation_pose(ann);xyz=points@R.T+t.reshape(3)/1000
    if not np.isfinite(xyz).all():raise ValueError('nonfinite camera coordinates')
    if np.any(xyz[labelled,2]<=0):raise ValueError('labelled landmark behind camera')
    q=xyz[labelled]@legacy.intrinsics(ann).T;uv=q[:,:2]/q[:,2:]
    if not np.isfinite(uv).all():raise ValueError('nonfinite labelled projection')
    error=float(np.max(np.linalg.norm(uv-stored[labelled,:2],axis=1)))
    ignored=np.flatnonzero((~labelled)&(xyz[:,2]<=0)).tolist()
    return error,ignored


@contextmanager
def preparation_checker(audit):
    original=legacy.reprojection_error
    def checked(ann,points):
        error,ignored=visibility_reprojection(ann,points)
        audit['annotations_checked']+=1
        if ignored:audit['unlabelled_nonpositive_cases'].append({'annotation_id':ann['id'],'image_id':ann['image_id'],'bop':ann.get('bop'),'indices':ignored})
        return error
    legacy.reprojection_error=checked
    try:yield
    finally:legacy.reprojection_error=original


def prepare_object(source_record,inventory,out,smoke=False):
    audit={'policy':'only COCO v=0 is exempt from positive-depth gate; labelled points remain strict',
           'annotations_checked':0,'unlabelled_nonpositive_cases':[],
           'frozen_adapter_sha256':legacy.sha(legacy.__file__),
           'preparation_checker_sha256':legacy.sha(__file__),
           'poses_RGB_masks_boxes_keypoint_order_and_membership_changed':False}
    with preparation_checker(audit):
        result=legacy.prepare_object(source_record,inventory,out,smoke=smoke)
    out=Path(out);legacy.save(out/'PREPARATION_AUDIT.json',audit)
    result['preparation_compatibility']={'audit_sha256':legacy.sha(out/'PREPARATION_AUDIT.json'),
                                       'checker_sha256':legacy.sha(__file__)}
    from hccepose_label_clipping import recipe
    result['render_compatibility']=recipe()
    legacy.save(out/'INPUTS_READY.json',result)
    return result
