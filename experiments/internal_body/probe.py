"""A frozen-state readout for the numeric health text, independent of actor fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.direct_policy.protocol import rng,write
from experiments.internal_body.design import prompt_levels


def run(cache,out):
    tr=np.arange(768);te=np.arange(768,960)
    groups=((1.25,2.5),(120.,350.),(2.,3.),(90.,300.))
    meta=json.loads((cache/"prompt-index.json").read_bytes())
    plain=np.load(cache/"native-state.npz")
    arrays={k:[] for k in ("residual","down_features","native_scores","token_length","no_health_residual")}
    for gi,levels in enumerate(groups):
        indices=tr if gi<2 else te;combined={k:[] for k in arrays}
        for h in levels:
            j=prompt_levels().index(h)
            record=meta[j]
            with np.load(cache/f"prompt-{j:02}.npz") as z:
                pos=np.searchsorted(z["index"],indices)
                assert record["health"]==h and np.array_equal(z["index"][pos],indices)
                combined["residual"].append(z["r"][pos].astype(float))
                combined["down_features"].append(z["features"][pos].astype(float))
                combined["native_scores"].append(z["native"][pos].astype(float))
                combined["no_health_residual"].append(plain["r"][indices].astype(float))
                combined["token_length"].append(np.array([len(record["ids"][i]) for i in pos],float)[:,None])
        for k in arrays:arrays[k].append(np.stack(combined[k]))
    gen=rng(64805,"shuffled-numeric-readouts")
    signs=np.column_stack((np.ones(len(tr)),gen.choice([-1.,1.],size=(len(tr),32))))
    result={}
    for k,(lo,hi,lot,hit) in arrays.items():
        diff=hi.mean(0)-lo.mean(0)
        directions=diff.T@signs/len(tr)
        center=(hi.mean((0,1))+lo.mean((0,1)))/2
        low=(lot-center)@directions
        high=(hit-center)@directions
        def positive(x):return (x>0).astype(float)+.5*(x==0)
        acc=(positive(-low).mean((0,1))+positive(high).mean((0,1)))/2
        order=positive(high[:,None]-low[None,:]).mean((0,1,2))
        result[k]=dict(real_balanced_accuracy=float(acc[0]),real_matched_ranking=float(order[0]),
                       shuffled_accuracy=acc[1:].tolist(),shuffled_ranking=order[1:].tolist(),
                       train_pair_direction_l2=float(np.linalg.norm(directions[:,0])))
    write(out,result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=run(a.cache,a.output)
    print(json.dumps({k:[v["real_balanced_accuracy"],v["real_matched_ranking"]] for k,v in r.items()}))
