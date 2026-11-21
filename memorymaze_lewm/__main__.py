import argparse
from pathlib import Path

from .io import read_json,write_json


def main():
    parser=argparse.ArgumentParser(description='Memory Maze offline LeWM pipeline')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('acquire');p.add_argument('--root',default='data');p.add_argument('--train',type=int,default=160);p.add_argument('--eval',type=int,default=30);p.add_argument('--eval-skip',type=int,default=10);p.add_argument('--workers',type=int,default=4)
    p=sub.add_parser('prepare');p.add_argument('--root',default='data');p.add_argument('--scaled-protocol',action='store_true');p.add_argument('--model-train',type=int,choices=[500,2000],default=500)
    p=sub.add_parser('validate');p.add_argument('--root',default='data');p.add_argument('--reference')
    p=sub.add_parser('pretrained');p.add_argument('--out',default='data/pretrained/cube')
    p=sub.add_parser('segments');p.add_argument('--splits',default='data/splits.json');p.add_argument('--out',default='data/segments_candidates.json');p.add_argument('--max-per-episode',type=int,default=1);p.add_argument('--max-step',type=int,default=50)
    p=sub.add_parser('review');p.add_argument('--segments',default='data/segments_candidates.json');p.add_argument('--decisions',required=True);p.add_argument('--out',default='data/segments_reviewed.json')
    p=sub.add_parser('train');p.add_argument('--config',default='configs/pilot.json');p.add_argument('--splits',default='data/splits.json');p.add_argument('--run',required=True);p.add_argument('--pretrained');p.add_argument('--resume',action='store_true');p.add_argument('--model-factory');p.add_argument('--stop-after',type=int);p.add_argument('--init-checkpoint')
    p=sub.add_parser('rollout');p.add_argument('--checkpoint',required=True);p.add_argument('--segments',default='data/segments_reviewed.json');p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--factory');p.add_argument('--allow-unreviewed',action='store_true');p.add_argument('--max-step',type=int,default=50);p.add_argument('--threads',type=int,default=3)
    p=sub.add_parser('probe');p.add_argument('--latents',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('diagnose');p.add_argument('--checkpoint',required=True);p.add_argument('--segments',default='data/segments_reviewed.json');p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p=sub.add_parser('export');p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='acquire':
        from .acquire import acquire
        acquire(args.root,args.train,args.eval,args.eval_skip,args.workers)
    elif args.command=='prepare':
        from .data import make_splits
        if args.scaled_protocol:
            expected=read_json(Path(__file__).resolve().parents[1]/'protocol/expected_membership.json')[str(args.model_train)]
            result=make_splits(args.root,counts=(args.model_train,100,300,100,200),protocol=True,expected_episode_ids=expected)
        else:result=make_splits(args.root)
        print(result['counts'])
    elif args.command=='validate':
        from .validation import validate_data
        print(validate_data(args.root,args.reference))
    elif args.command=='pretrained':
        from .pretrained import download_cube
        print(download_cube(args.out))
    elif args.command=='segments':
        from .segments import build_segments
        result=build_segments(args.splits,args.out,args.max_per_episode,args.max_step)
        print('Candidates:',len(result['segments']),'without candidate:',len(result['rejected']))
    elif args.command=='review':
        result=read_json(args.segments);decisions=read_json(args.decisions)
        for seg in result['segments']:
            decision=decisions.get(seg['segment_id'])
            if decision:
                if decision['status'] not in ['approved','rejected']:raise ValueError('Invalid review decision')
                seg.update(review_status=decision['status'],review_note=decision['note'],reviewer=decision.get('reviewer','inherited recorded review'))
        result['status']='initial-context review recorded; future visibility remains heuristic unless separately annotated'
        write_json(args.out,result)
    elif args.command=='train':
        from .train import train
        train(args.config,args.splits,args.run,args.pretrained,args.resume,args.model_factory,args.stop_after,args.init_checkpoint)
    elif args.command=='rollout':
        from .rollout import extract
        extract(args.checkpoint,args.segments,args.out,args.device,args.factory,args.allow_unreviewed,args.max_step,args.threads)
    elif args.command=='probe':
        from .evaluate import run_probe
        result=run_probe(args.latents,args.out);print(result['selected_lambda'])
    elif args.command=='diagnose':
        from .diagnose import diagnose
        print(diagnose(args.checkpoint,args.segments,args.out,args.device))
    elif args.command=='export':
        import torch
        from .train import save_checkpoint
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        keys=['model','model_config','config','action_stats','initialization','update']
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise FileExistsError(out)
        save_checkpoint(out,{k:checkpoint[k] for k in keys});print(out)


if __name__=='__main__':main()
