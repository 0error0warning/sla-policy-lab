"""Post-fit diagnostics on the original, unrestricted single-life learner."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.consequence_policy.study import PARTNERS
from experiments.direct_policy.protocol import probabilities,write
from experiments.internal_body.design import PILOT
from experiments.internal_body.lives import WORLDS
from experiments.internal_body.personal import LifeInputs


def diagnostics(cache,out):
    data=LifeInputs(cache)
    look=lambda t,w,f:np.array([i for i,r in enumerate(data.rows) if r["split"]=="new_partner" and
                                r["task"]==t and r["wording"]==w and r["frame"]==f])
    kernels={}
    for name,keys in (("request_only",data.semantic),("decision",data.decision)):
        kernels[name]=[[float((keys[look(4,3,i)]@keys[look(5,4,j)].T).mean()) for j in (0,1)] for i in (0,1)]
    groups=[];match=[]
    for seed in PILOT:
        meta=json.loads((out/f"life-{seed}.json").read_bytes())
        with np.load(out/f"life-{seed}-raw.npz") as z:
            for j,world in enumerate(WORLDS):
                diff=probabilities(PARTNERS[0],0,world[0])[1]!=probabilities(PARTNERS[1],0,world[1])[1]
                row=next(v["arms"] for v in meta if not v["reversal"] and tuple(v["world"])==world)
                match.append(dict(seed=seed,world=world,actually_opposite=diff,
                    fixed=row["fixed"]["post"]["style"],free=row["free"]["post"]["style"],tied=row["tied"]["post"]["style"],
                    seen=row["free"]["seen"]["style"],swapped=row["free"]["exchanged"]["style"]))
                tr=z[f"w{j}-v0-free"]
                for p in range(2):
                    for f in range(2):
                        a=np.array([v for v in tr if data.rows[int(v[0])]["partner"]%2==p and data.rows[int(v[0])]["frame"]==f and v[1]>=15.])
                        best=2 if probabilities(PARTNERS[p],f,world[p])[1]>.5 else 3
                        if not len(a):continue
                        groups.append(dict(seed=seed,world=world,partner=p,frame=f,n=len(a),
                            both_styles=int((a[:,2]==2).any() and (a[:,2]==3).any()),
                            correct_offers=int((a[:,2]==best).sum()),correct_accepted=int(((a[:,2]==best)&(a[:,3]==1)).sum()),
                            first_correct_probability=float(a[0,7+best]),last_correct_probability=float(a[-1,7+best]),
                            any_correct=int((a[:,2]==best).any()),gradient=float(a[:,-2].mean()),fisher=float(a[:,-1].mean())))
    results=dict(language_kernel=kernels,per_group=groups,per_world=match,
       both_styles_fraction=float(np.mean([r["both_styles"] for r in groups])),
       any_correct_fraction=float(np.mean([r["any_correct"] for r in groups])),
       first_correct_prob=float(np.mean([r["first_correct_probability"] for r in groups])),
       last_correct_prob=float(np.mean([r["last_correct_probability"] for r in groups])),
       actual_preferences={str(k):{a:float(np.mean([r[a] for r in match if r["actually_opposite"]==k]))
                                   for a in ("fixed","free","tied","seen","swapped")} for k in (False,True)})
    write(out/"personal-diagnostic.json",results)
    print(json.dumps({k:v for k,v in results.items() if k not in ("per_group","per_world")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();diagnostics(a.cache,a.output)
