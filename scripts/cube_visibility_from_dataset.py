"""Cube pixels (red, base and wrist camera) at chosen frames of the 150 v3 episodes that have a known spawn (collectors a and b, 75 each).
Reads the images stored in the dataset (no simulator needed); the episode order of ur5e_pick_place_v3 = a then b = the collectors' log order.
Output: eval_results/v3_visibility_by_frame.npz. Findings: README 2026-11-12 (cube visibility).
"""
import glob, io, re, json, csv, numpy as np, pandas as pd
from PIL import Image
FR=[0,2,10,25,40,50,60,75,100,125,150,200,300]
def px(img):
    a=np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB")).astype(np.int16)
    return int(((a[...,0]-np.maximum(a[...,1],a[...,2]))>40).sum())
P=[];B={f:[] for f in FR};W={f:[] for f in FR}
for n in "ab":
    root=f"/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/hf_cache/vla_ur5e_ws/v3_wide_{n}"
    df=pd.concat([pd.read_parquet(f,columns=["episode_index","frame_index","image","wrist_image"]) for f in sorted(glob.glob(root+"/data/chunk-*/*.parquet"))])
    df=df[df["frame_index"].isin(FR)].sort_values(["episode_index","frame_index"])
    t=open(f"eval_runs/collect/wide_{n}.log").read()
    pos=[[float(v) for v in m.group(1).split()[:2]] for m in re.finditer(r"episode \d+/75: OK.*?cube spawned at \[([-\d. ]+)\]",t)]
    P+=pos
    for ep,g in df.groupby("episode_index"):
        g=g.set_index("frame_index")
        for f in FR:
            B[f].append(px(g.loc[f,"image"])); W[f].append(px(g.loc[f,"wrist_image"]))
P=np.array(P); B={f:np.array(v) for f,v in B.items()}; W={f:np.array(v) for f,v in W.items()}
np.savez("eval_results/v3_visibility_by_frame.npz",pos=P,frames=np.array(FR),**{f"base_{f}":B[f] for f in FR},**{f"wrist_{f}":W[f] for f in FR})
print("frame | base: cube visible (>=20 px) in n/150, median px | wrist: visible n/150")
for f in FR: print(f"{f:5d} | {int((B[f]>=20).sum()):3d}/150 median {int(np.median(B[f])):4d} | {int((W[f]>=20).sum()):3d}/150")
