#!/usr/bin/env python3
"""Reproduce a paired experiment without opening a browser."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
from pathlib import Path
from sla_demo.environment import Config
from sla_demo.models import ModelSuite
from sla_demo.experiment import run
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile',choices=['balanced','service','preserve'],default='balanced')
parser.add_argument('--partner',choices=['supportive','mixed','unavailable'],default='mixed')
parser.add_argument('--horizon',type=int,default=60)
parser.add_argument('--load',type=float,default=.65)
parser.add_argument('--initial-crack',type=float,default=.08)
parser.add_argument('--seed-start',type=int,default=100)
parser.add_argument('--seed-count',type=int,default=24)
parser.add_argument('--train-seed',type=int,default=17)
parser.add_argument('--strength',type=float,default=1)
parser.add_argument('--output',type=Path,default=Path('results.json'))
args=parser.parse_args()
try:
    if not 0<=args.train_seed<=2**31-10000:raise ValueError('train-seed out of range')
    config=Config(horizon=args.horizon,load=args.load,initial_crack=args.initial_crack,partner=args.partner,profile=args.profile)
    # Validate experiment controls before expensive training.
    if not 1<=args.seed_count<=128:raise ValueError('seed-count must be 1..128')
    suite=ModelSuite(args.train_seed)
    result=run(suite,config,args.seed_start,args.seed_count,args.strength)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(f'Saved {args.output.resolve()}')
    print(f'Model fingerprint: {suite.fingerprint}')
    for key,value in result['methods'].items():
        s=value['summary'];print(f"{key:18} return={s['return']['mean']:7.3f}  lifespan={s['lifespan']['mean']:6.2f}  answers={s['answers']['mean']:6.2f}")
except ValueError as error:parser.error(str(error))
