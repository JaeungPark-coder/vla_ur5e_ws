#!/usr/bin/env python3
"""Expert side of scripts/corner_view.py (GPU-free): red-cube pixels and wrist-image position of the cube in the v3 DATASET images at frames
50-200, by region (grip point at the first close as the cube-position proxy), then a montage expert (corner episodes) vs v3 6k (corner trials)
at ticks 50/100 -> eval_results/corner_expert_vs_6k.png.  python3 scripts/corner_expert_view.py"""
import sys, glob, io, os,numpy as np,pandas as pd
from PIL import Image, ImageDraw
sys.path.insert(0,'scripts')
from ur5e_fk import apply
FK_P=np.array([-np.pi,0,0,0,0,0.12,0])
def mask(im):
    im=im[...,:3].astype(int); return (im[...,0]>120)&(im[...,0]>im[...,1]+60)&(im[...,0]>im[...,2]+60)
def dec(v): return np.asarray(Image.open(io.BytesIO(v['bytes'])).convert('RGB'))
R={'+Y corner':lambda x,y:x<0.42 and y>0.11,'-Y corner':lambda x,y:x<0.42 and y<-0.11,'X<0.42 mid':lambda x,y:x<0.42 and abs(y)<=0.11,'+Y edge X>=.42':lambda x,y:x>=0.42 and y>0.10,'-Y edge X>=.42':lambda x,y:x>=0.42 and y<-0.10}
stats={r:{t:[] for t in (50,100,150,200)} for r in R}; base={r:{t:[] for t in (50,100,150,200)} for r in R}
keep=[]
for f in sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/lerobot/Jaeung12/ur5e_pick_place_v3/data/*/*.parquet'))):
    d=pd.read_parquet(f)
    for ep,dd in d.groupby('episode_index'):
        q=np.stack(dd['joints'].values)[:,:6].astype(float); g=np.concatenate([np.atleast_1d(x) for x in dd['gripper'].values])
        c=np.nonzero(g>0.5)[0]
        if not len(c): continue
        gp=apply(FK_P,q); x,y=gp[c[0],0],gp[c[0],1]
        for r,fn in R.items():
            if fn(x,y):
                for t in (50,100,150,200):
                    row=dd[dd.frame_index==t]
                    if not len(row): continue
                    m=mask(dec(row['wrist_image'].iloc[0])); mb=mask(dec(row['image'].iloc[0]))
                    base[r][t].append(int(mb.sum()))
                    if m.sum():
                        rr,cc=np.nonzero(m); stats[r][t].append((cc.mean(),rr.mean(),m.sum(),bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())))
                    else: stats[r][t].append(None)
                if r=='+Y corner': keep.append((ep,x,y))
for r in R:
    print(r, 'episodes', len(base[r][100]))
    for t in (50,100,150,200):
        s=[v for v in stats[r][t] if v]; n=len(stats[r][t])
        if n==0: continue
        print(f"   t{t:3d} wrist seen {len(s)}/{n}", (f"col {np.median([v[0] for v in s]):.0f} row {np.median([v[1] for v in s]):.0f} px {np.median([v[2] for v in s]):.0f} border {sum(v[3] for v in s)}/{len(s)}" if s else ""), f"| base px median {np.median(base[r][t]):.0f}, zero {sum(b==0 for b in base[r][t])}/{n}")
print('corner episodes', [(e,round(x,3),round(y,3)) for e,x,y in keep])

# ---- montage
from PIL import Image, ImageDraw
eps=[154,188,211,224]; trials=[(8,'6k 8007'),(13,'6k 8012'),(20,'6k 8019'),(34,'6k 8033')]
S=192; ticks=(50,100)
rows=[]
for f in sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/lerobot/Jaeung12/ur5e_pick_place_v3/data/*/*.parquet'))):
    d=pd.read_parquet(f,columns=['image','wrist_image','frame_index','episode_index']); d=d[d.episode_index.isin(eps)&d.frame_index.isin(ticks)]
    for ep in eps:
        dd=d[d.episode_index==ep]
        if len(dd)==len(ticks): rows.append((f'expert ep{ep}',[np.asarray(Image.open(io.BytesIO(dd[dd.frame_index==t][k].iloc[0]['bytes'])).convert('RGB')) for k in ('image','wrist_image') for t in ticks]))
for t,lab in trials:
    z=np.load(f'eval_runs/n60_v3_6k/trace/trial_{t:03d}_images.npz'); T=list(z['tick'])
    rows.append((lab,[z[k][T.index(tt)] for k in ('base','wrist') for tt in ticks]))
L=110; c=Image.new('RGB',(L+4*S,len(rows)*S+16),'white'); dr=ImageDraw.Draw(c)
for j,h in enumerate(['base t50','base t100','wrist t50','wrist t100']): dr.text((L+j*S+4,2),h,fill='black')
for i,(lab,ims) in enumerate(rows):
    dr.text((4,16+i*S+S//2),lab,fill='black')
    for j,im in enumerate(ims): c.paste(Image.fromarray(np.asarray(im)).resize((S,S)),(L+j*S,16+i*S))
c.save('eval_results/corner_expert_vs_6k.png'); print('saved',len(rows))
