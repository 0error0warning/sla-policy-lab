"""Post-hoc action-only learning control, without partner/framing conditioning."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import independent_body
from core import (CONFIG, BodyValue, Evidence, execute_event, schedule,
                  true_probabilities, write)


def pooled(evidence):
    counts = evidence.counts.sum(axis=(0,1))
    return (counts[:,0]+1.)/(counts.sum(axis=1)+2.)


def run(raw,output):
    output.mkdir(parents=True,exist_ok=True)
    body = BodyValue()
    results,replays = [],[]
    for seed in CONFIG["seeds"]:
        for mapping in CONFIG["mappings"]:
            evidence,health = Evidence(),350.
            acting = np.full(3,.5)
            trace,regrets = [],[]
            for item in schedule(seed):
                record = execute_event(body,evidence,health,item,mapping,seed,acting)
                health = record["health_after"]
                assert max(abs(a-b) for a,b in zip(independent_body(record["health_before"],record["action"],record["accepted"],record["noise"]),record["body_events"]))<1e-10
                values = body.values(record["health_before"],record["true_probabilities"])
                regrets.append(float(values.max()-values[record["action"]]))
                trace.append(record)
                if item["turn"]%5==0:
                    acting = pooled(evidence)
                if health<=0:
                    break
            prediction = pooled(evidence)
            truth = [true_probabilities(p,f,mapping) for p in ("P17","P42") for f in range(2)]
            result = dict(seed=seed,mapping=mapping,method="pooled",mode="posthoc_closed",trace=trace,
                final_health=health,turns=len(trace),alive_at_end=bool(health>0),
                total_answers=sum(r["action"]>=2 for r in trace),
                total_protected=sum(r["accepted"] is True for r in trace),
                final_truth_mae=float(np.abs(np.array(truth)-prediction).mean()),
                final_person_binding_margin=0.,mean_one_step_oracle_regret=float(np.mean(regrets)),
                final_probabilities=prediction.tolist())
            write(output/f"closed-s{seed}-m{mapping}.json",result)
            results.append({k:v for k,v in result.items() if k!="trace"})
            if seed in CONFIG["replay_seeds"]:
                source = json.loads((raw/f"output/closed-s{seed}-m{mapping}-hierarchical/result.json").read_bytes())
                e = Evidence()
                e.counts[:] = source["trace"][-1]["evidence_after"]["counts"]
                p = pooled(e)
                replays.append(dict(seed=seed,mapping=mapping,
                    final_truth_mae=float(np.abs(np.array(truth)-p).mean()),probabilities=p.tolist()))
    fields=("final_health","total_answers","total_protected","final_truth_mae","mean_one_step_oracle_regret")
    result = dict(post_hoc=True,methods="Beta(1,1) per action; no partner or framing in the prediction",
        scope="same body, five-turn refresh and epsilon=.2; adds a calibration-only baseline",
        n=len(results),aggregate={k:float(np.mean([r[k] for r in results])) for k in fields},
        alive_at_end=sum(r["alive_at_end"] for r in results),
        matched_history_truth_mae=float(np.mean([r["final_truth_mae"] for r in replays])),results=results,replays=replays)
    write(output/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("results","replays")}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("raw",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.raw,args.output)
