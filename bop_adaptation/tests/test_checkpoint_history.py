import sys
import types
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from hccepose_train import loss_state,restore_loss


def test_adaptive_loss_history_preserves_numpy_dtype(monkeypatch):
    class FixedSizeList:
        def __init__(self,n):self.data=[]
        def append(self,v):self.data.append(v)
    module=types.ModuleType('HccePose.network_model');module.FixedSizeList=FixedSizeList
    monkeypatch.setitem(sys.modules,'HccePose',types.ModuleType('HccePose'))
    monkeypatch.setitem(sys.modules,'HccePose.network_model',module)
    for dtype in [np.float32,np.float64]:
        original=types.SimpleNamespace();restored=types.SimpleNamespace()
        for side in ['Front','Back']:
            histories=[[FixedSizeList(100) for bit in range(8)] for axis in range(3)]
            for axis in histories:
                for bit in axis:bit.append(np.asarray(.12345,dtype=dtype))
            setattr(original,side+'_error_list',histories)
            setattr(original,'weight_'+side.lower()+'_error_ratio',[torch.ones(8,dtype=torch.float64) for _ in range(3)])
        state=loss_state(original);restore_loss(restored,state,'cpu')
        for side in ['Front','Back']:
            before=getattr(original,side+'_error_list')[0][0].data[0];after=getattr(restored,side+'_error_list')[0][0].data[0]
            assert before.dtype==after.dtype and np.array_equal(before,after)
            assert before is not after
