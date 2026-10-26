#!/usr/bin/env python3
"""Pin and convert an existing per-object frozen COCO/geometry export.

This is the adapter used for the BOP-derived experimental populations, not a
new raw-BOP scene sampler. It never rewrites source images, masks or geometry.
"""
import argparse
import json
from pathlib import Path
from hccepose_data import sha,save
from hccepose_prepare_visibility import prepare_object


def inventory(root,dataset,obj):
    root=root.resolve();keypoints=root/'geometry/keypoints_3d.json';splits={}
    for split in ['train','val']:
        path=root/'dataset'/split/'annotations.json';coco=json.loads(path.read_text())
        splits[split]={'images':len(coco['images']),'annotations':len(coco['annotations']),'sha256':sha(path)}
    return {'dataset':dataset,'object':obj,'base_root':str(root),'keypoints':str(keypoints),
            'keypoints_sha256':sha(keypoints),'splits':splits}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    inv=sub.add_parser('inventory');inv.add_argument('--source-root',type=Path,required=True);inv.add_argument('--dataset',required=True);inv.add_argument('--object',type=int,required=True);inv.add_argument('--out',type=Path,required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--source',type=Path,required=True);prep.add_argument('--out',type=Path,required=True);prep.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.mode=='inventory':
        if a.out.exists():raise FileExistsError('inventory already exists; do not silently repin inputs')
        save(a.out,inventory(a.source_root,a.dataset,a.object))
    else:
        source=json.loads(a.source.read_text());prepare_object(source,source,a.out,smoke=a.smoke)
