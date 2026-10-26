from pathlib import Path
import sys
import types
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import hccepose_label_clipping as clipping


def test_clipping_only_changes_nonpositive_near_plane():
    assert clipping.clipping_planes([10,30])==(10.,30.,False)
    near,far,changed=clipping.clipping_planes([-20,300])
    assert changed and 0<near<far and near<=1 and far==300
    with pytest.raises(ValueError,match='behind'):clipping.clipping_planes([-20,0])
    with pytest.raises(ValueError,match='invalid'):clipping.clipping_planes([1,float('nan')])


def test_regular_render_delegates_exact_arguments_and_crossing_render_keeps_pose(tmp_path,monkeypatch):
    source=tmp_path/'renderer.py';source.write_text('fixture');monkeypatch.setattr(clipping,'UPSTREAM',source)
    projections=[]
    module=types.ModuleType('clip_test_renderer');module.__file__=str(source)
    module._calc_calib_proj=lambda *args:projections.append(args) or np.eye(4)
    monkeypatch.setitem(sys.modules,'clip_test_renderer',module)
    class Renderer:
        def __init__(self):
            self.model_bbox_corners={0:np.array([[0,0,-1],[1,1,2]])};self.pose_cv_to_gl=np.diag([1,-1,-1,1]);self.width=8;self.height=8;self.mode='depth';self.depth=np.ones((8,8));self.calls=[]
        def render_object(self,*args,**kwargs):self.calls.append((args,kwargs));return {'legacy':True}
        def update(self):pass
        def on_draw(self,obj,model,view,projection,**kwargs):self.draw=(obj,model,view,projection,kwargs)
    Renderer.__module__='clip_test_renderer'
    backend=types.SimpleNamespace(renderer=types.SimpleNamespace(create_renderer=lambda *args,**kwargs:Renderer()))
    monkeypatch.setattr(clipping,'backend',lambda:backend)
    stats=clipping.install(clipping.recipe());renderer=backend.renderer.create_renderer(8,8,'vispy');R=np.eye(3);t=np.array([[0],[0],[10.]])
    assert renderer.render_object(0,R,t,500,500,4,4,clear=False,draw_back=True)=={'legacy':True}
    assert renderer.calls[0][0][1] is R and renderer.calls[0][0][2] is t
    assert renderer.calls[0][1]=={'clear':False,'draw_back':True}
    result=renderer.render_object(0,R,np.zeros((3,1)),500,500,4,4)
    assert 'depth' in result and len(renderer.calls)==1
    assert stats['renders']==2 and stats['corrected_renders']==1
    assert projections[-1][-2:]==(renderer.clip_near,renderer.clip_far)
    np.testing.assert_array_equal(renderer.draw[1],np.eye(4))
    np.testing.assert_array_equal(renderer.draw[2],renderer.pose_cv_to_gl.T)
