"""Conservative red-object candidates; human review is required for final claims."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import HORIZONS, cached_episode
from .io import read_json, write_json


def red_id(d):
    color=d['target_color'];red=(color[:,0]>1.8*color[:,1])&(color[:,0]>1.8*color[:,2])
    if red.any():
        distance=np.linalg.norm(d['targets_pos'][red]-d['target_pos'][red,None,:],axis=-1)
        indices=distance.argmin(-1)
        if np.any(indices!=0) or np.any(distance[np.arange(len(indices)),indices]>1e-5):
            raise ValueError('Color/object ID mapping differs from default official red ID 0')
    # Official maze.py TARGET_COLORS[0] is red; randomize_colors defaults False.
    return 0


def red_pixels(images):
    a=images[:,2:-2,2:-2].astype(np.float32)
    mask=(a[...,0]>60)&(a[...,0]>1.8*a[...,1])&(a[...,0]>1.8*a[...,2])
    return mask.sum((1,2))


def line_of_sight(d,t,object_id):
    start=d['agent_pos'][t];end=d['targets_pos'][t,object_id]
    points=np.linspace(start,end,max(2,int(np.linalg.norm(end-start)*30)))
    cells=np.floor(points).astype(int);maze=d['maze_layout'][t]
    inside=(cells[:,0]>=0)&(cells[:,0]<maze.shape[1])&(cells[:,1]>=0)&(cells[:,1]<maze.shape[0])
    return bool(inside.all() and (maze[cells[:,1],cells[:,0]]>0).all())


def select_candidates(d,max_segments=1,max_step=50):
    if max_step<5:raise ValueError('max_step must be at least 5')
    object_id=red_id(d);count=red_pixels(d['image']);n=len(count)
    vec=d['targets_vec'][:,object_id]
    # Axis convention validated from agent_dir/delta in acquisition QA: second is forward.
    front=(vec[:,1]>1.2)&(np.abs(vec[:,0])<vec[:,1]*.45)
    visible=(count>=12)&front
    candidates=[];next_start=2
    for origin in range(2,n-max_step):
        if origin<next_start or not visible[origin-2:origin+1].all():continue
        if d['reset'][origin-1:origin+max_step+1].any() or d['terminal'][origin-2:origin+max_step].any():continue
        if not all(line_of_sight(d,t,object_id) for t in range(origin-2,origin+1)):continue
        future_hidden=count[origin+1:origin+max_step+1]==0
        if not any(future_hidden[i:i+5].all() for i in range(max_step-4)):continue
        candidates.append({'origin':origin,'object_id':object_id,'red_pixels_initial':count[origin-2:origin+1].tolist(),
                           'visibility':['visible' if x>=12 else 'hidden' if x==0 else 'partial' for x in count[origin:origin+max_step+1]],
                           'first_hidden_step':int(np.flatnonzero(future_hidden)[0]+1),
                           'review_status':'pending','visibility_source':'red pixel heuristic, not authoritative labels'})
        next_start=origin+max_step+3
        if len(candidates)>=max_segments:break
    return candidates


def build_segments(splits_path,out_path,max_segments=1,max_step=50):
    splits=read_json(splits_path)['splits'];records=[];rejected=[]
    for split in ['probe_train','probe_val','test']:
        for path in splits[split]:
            d=cached_episode(path);items=select_candidates(d,max_segments,max_step)
            if not items:rejected.append({'path':path,'split':split,'reason':'no candidate met initial/hidden/length rules'})
            for i,item in enumerate(items):
                item.update({'path':path,'episode_id':Path(path).stem,'segment_id':f'{Path(path).stem}:{item["origin"]}',
                             'split':split,'horizons':list(range(max_step+1))})
                records.append(item)
    result={'segments':records,'rejected':rejected,'status':'candidate generation; initial and visibility review pending',
            'id_mapping_source':'https://github.com/jurgisp/memory-maze/blob/main/memory_maze/maze.py TARGET_COLORS[0]; validate observed target_color/target_pos'}
    write_json(out_path,result);contact_sheets(records,Path(out_path).parent/'review')
    return result


def contact_sheets(records,out):
    out.mkdir(parents=True,exist_ok=True)
    for begin in range(0,len(records),12):
        batch=records[begin:begin+12];canvas=Image.new('RGB',(768,110*len(batch)), 'white');draw=ImageDraw.Draw(canvas)
        for row,seg in enumerate(batch):
            d=cached_episode(seg['path']);t=seg['origin']
            last=seg['horizons'][-1]
            frames=[t-2,t-1,t,t+seg['first_hidden_step'],t+min(10,last),t+last]
            draw.text((4,row*110),f'{begin+row}: {seg["episode_id"]} t0={t} ID={seg["object_id"]} {seg["split"]}',fill='black')
            for col,index in enumerate(frames):
                image=Image.fromarray(d['image'][index]).resize((96,80),Image.Resampling.NEAREST)
                canvas.paste(image,(col*128,row*110+26));draw.text((col*128,row*110+13),f't={index}',fill='black')
        canvas.save(out/f'candidates_{begin//12:02d}.png')
