import numpy as np
import pytest
from hccepose_data import annotation_pose, intrinsics, reprojection_error, symmetry_matrices, OriginalPoseDepthRenderer, StrictCV2, CV_TO_GL


def test_frozen_coco_to_original_bop_frame_units():
    camera=np.eye(4);camera[:3,3]=[.1,.2,.3]
    expected=np.eye(4);expected[:3,3]=[.02,-.01,.8]
    obj=camera@CV_TO_GL@expected
    points=np.array([[0,0,0],[.01,.02,.03]])
    K=np.array([[500,0,320],[0,510,240],[0,0,1]])
    xyz=points+expected[:3,3];h=xyz@K.T;uv=h[:,:2]/h[:,2:]
    ann={'camera_extrinsics':camera.tolist(),'object_pose':obj.tolist(),
         'camera_intrinsics':{'fx':500,'fy':510,'cx':320,'cy':240},
         'keypoints':np.c_[uv,np.ones(2)].ravel().tolist()}
    R,t=annotation_pose(ann)
    np.testing.assert_allclose(R,np.eye(3));np.testing.assert_allclose(t.ravel(),[20,-10,800])
    assert reprojection_error(ann,points)<1e-9
    del ann['camera_intrinsics']['fx']
    with pytest.raises(KeyError):intrinsics(ann)


def test_symmetry_translation_and_identity_are_not_discarded():
    s={'R':np.diag([-1,-1,1]),'t':np.array([[20],[30],[0]])}
    out=symmetry_matrices([s,np.eye(4).ravel().tolist()])
    np.testing.assert_allclose(np.asarray(out[0]).reshape(4,4)[:3,3],[20,30,0])
    np.testing.assert_allclose(np.asarray(out[1]).reshape(4,4),np.eye(4))


def test_back_depth_uses_original_pose_not_mixed_canonical_rotation_translation():
    calls=[]
    class Fake:
        def render_object(self,obj,R,t,*args,**kwargs):calls.append((R.copy(),t.copy(),kwargs['draw_back']))
    p=OriginalPoseDepthRenderer(Fake());R=np.eye(3);t=np.array([[3],[5],[800]])
    p.render_object(0,R,t,500,500,320,240,draw_back=False)
    p.render_object(0,np.diag([-1,-1,1]),t+[[20],[30],[0]],500,500,320,240,draw_back=True)
    np.testing.assert_array_equal(calls[1][0],R);np.testing.assert_array_equal(calls[1][1],t)
    assert calls[1][2]


def test_missing_coordinate_labels_never_become_zero_supervision():
    class Fake:
        def imread(self,*a,**k):return None
        def imwrite(self,*a,**k):return False
    strict=StrictCV2(Fake())
    with pytest.raises(FileNotFoundError):strict.imread('missing_front.png')
    with pytest.raises(OSError):strict.imwrite('unwritable.png',np.zeros((2,2,3)))
