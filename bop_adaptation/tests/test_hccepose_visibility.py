from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import hccepose_data as legacy
from hccepose_prepare_visibility import visibility_reprojection,preparation_checker


def fixture():
    points=np.array([[0.,0.,-1.],[.01,.02,0.],[.02,.03,.01]])
    pose=np.eye(4);pose[2,3]=.5
    xyz=points+pose[:3,3];uv=xyz[:,:2]/xyz[:,2:]*500+320
    stored=np.c_[uv,[0,1,2]];stored[0]=[0,0,0]
    ann={'id':7,'image_id':9,'camera_extrinsics':np.eye(4).tolist(),
         'object_pose':(legacy.CV_TO_GL@pose).tolist(),
         'camera_intrinsics':{'fx':500,'fy':500,'cx':320,'cy':320},'keypoints':stored.ravel().tolist()}
    return ann,points


def test_unlabelled_nonpositive_is_not_an_annotation_error():
    ann,points=fixture();before=deepcopy(ann)
    with pytest.raises(ValueError,match='behind'):legacy.reprojection_error(ann,points)
    error,ignored=visibility_reprojection(ann,points)
    assert error<1e-9 and ignored==[0] and ann==before
    points[0,2]=-.5
    assert visibility_reprojection(ann,points)[1]==[0]  # unlabelled exactly at camera plane


@pytest.mark.parametrize('visibility',[1,2])
def test_all_labelled_points_still_require_positive_depth(visibility):
    ann,points=fixture();ann['keypoints'][2]=visibility
    with pytest.raises(ValueError,match='labelled landmark'):visibility_reprojection(ann,points)


def test_finite_checks_and_existing_projection_tolerance_remain():
    ann,points=fixture();ann['keypoints'][3]+=2
    assert visibility_reprojection(ann,points)[0]>1.9
    ann,points=fixture();points[0,0]=float('nan')
    with pytest.raises(ValueError,match='values'):visibility_reprojection(ann,points)


def test_preparation_only_override_restores_frozen_function_even_on_error():
    ann,points=fixture();original=legacy.reprojection_error;digest=legacy.sha(legacy.__file__)
    audit={'annotations_checked':0,'unlabelled_nonpositive_cases':[]}
    with pytest.raises(RuntimeError):
        with preparation_checker(audit):
            assert legacy.reprojection_error(ann,points)<1e-9
            raise RuntimeError('test failure')
    assert legacy.reprojection_error is original and legacy.sha(legacy.__file__)==digest
    assert audit['annotations_checked']==1 and audit['unlabelled_nonpositive_cases'][0]['indices']==[0]
