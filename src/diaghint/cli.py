"""Exactly four public commands; no experiment runners or alternative policies."""
import argparse
import json
from .config import load


def parser():
    p=argparse.ArgumentParser(prog='diaghint')
    p.add_argument('--config',default='config/diaghint.conf')
    commands=p.add_subparsers(dest='command',required=True)
    generate=commands.add_parser('generate-pool',help='Generate Stage-I and Stage-II experiences and exact training records')
    generate.add_argument('--workload',required=True,help='Directory containing training SQL only')
    generate.add_argument('--output',required=True)
    card=commands.add_parser('train-cardinality',help='Fit the fixed five-class cardinality-bias model')
    card.add_argument('--pool',required=True);card.add_argument('--output',required=True)
    utility=commands.add_parser('train-utility',help='Fit shared plan encoder and Demo/Reuse/offline-NoDemo value heads')
    utility.add_argument('--pool',required=True);utility.add_argument('--cardinality-model',required=True);utility.add_argument('--output',required=True)
    online=commands.add_parser('optimize',help='Optimize one read-only query using the fixed online route')
    online.add_argument('--sql',required=True);online.add_argument('--pool',required=True)
    online.add_argument('--cardinality-model',required=True);online.add_argument('--utility-model',required=True);online.add_argument('--output',required=True)
    return p


def main():
    args=parser().parse_args();config=load(args.config)
    if args.command=='generate-pool':
        from .pool import generate
        generate(config,args.workload,args.output)
    elif args.command=='train-cardinality':
        import torch
        torch.set_num_threads(1)
        from .learning import train_cardinality
        train_cardinality(config,args.pool,args.output)
    elif args.command=='train-utility':
        import torch
        torch.set_num_threads(1)
        from .learning import train_utility
        train_utility(config,args.pool,args.cardinality_model,args.output)
    else:
        import torch
        torch.set_num_threads(1)
        from .online import optimize
        print(json.dumps(optimize(config,args.sql,args.pool,args.cardinality_model,args.utility_model,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
