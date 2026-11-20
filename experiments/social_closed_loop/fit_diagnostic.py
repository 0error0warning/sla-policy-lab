"""Post-hoc CPU fitting diagnostic; does not replace closed-loop outcomes.

Ask whether the routed readout's residual error comes from fitting only the
latest question or from the representation/regularization. Teacher targets are
held fixed at the last actual update. No additional partner feedback is created.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core import CONFIG, Evidence, FastReadout, PARTNERS, features, sigmoid, write


def diagnose(raw, output):
    rows = json.loads((raw/"output/rows.json").read_bytes())
    baseline = json.loads((raw/"output/frozen.json").read_bytes())["log_odds"]
    hidden = np.load(raw/"output/frozen-hidden.npy",allow_pickle=False).astype(float)
    x,_ = features(hidden,rows)
    results = []
    original_ridge = CONFIG["fast_ridge"]
    for path in sorted((raw/"output").glob("*fast/result.json")):
        life = json.loads(path.read_bytes())
        evidence = Evidence()
        turn = life["turns"]//5*5
        evidence.counts[:] = life["trace"][turn-1]["evidence_after"]["counts"]
        evidence.observations = int(evidence.counts.sum())
        targets = evidence.target(rows)
        record = dict(seed=life["seed"],mapping=life["mapping"],mode=life["mode"],last_update=turn,variants={})
        try:
            for variant,ridge in (("saved",original_ridge),("all_questions",original_ridge),("all_questions_weaker_ridge",1e-7)):
                CONFIG["fast_ridge"] = ridge
                # Only the batch-selector metadata is changed; model prompts,
                # frozen features, labels and evaluation rows stay identical.
                fit_rows = [dict(row,task=0) if row["split"]=="train" else row for row in rows]
                readout = FastReadout(fit_rows,x,baseline)
                readout.w = {p:np.array(life["fast_weights"][p],dtype=float) for p in PARTNERS}
                if variant != "saved":
                    readout.update(evidence,0)
                else:
                    final = life["probes"][-1]
                    assert np.max(abs(readout.predict()[final["row_ids"]]-np.array(final["log_odds"]))) < 1e-7
                prediction = sigmoid(readout.predict())
                metrics = {}
                for split in ("train","heldout","paraphrase","content"):
                    ids = [i for i,r in enumerate(rows) if r["split"]==split and r["partner"]!="P86"]
                    metrics[split] = float(np.abs(prediction[ids]-targets[ids]).mean())
                record["variants"][variant] = dict(teacher_mae=metrics,ridge=ridge,
                    coefficient_l2=float(np.sqrt(sum(w@w for w in readout.w.values()))),
                    optimization=readout.optimization)
        finally:
            CONFIG["fast_ridge"] = original_ridge
        results.append(record)
    aggregate = {}
    for mode in ("closed","replay"):
        subset = [r for r in results if r["mode"]==mode]
        if not subset:
            continue
        aggregate[mode] = {variant:{split:float(np.mean([r["variants"][variant]["teacher_mae"][split] for r in subset]))
                                   for split in ("train","heldout","paraphrase","content")}
                           for variant in ("saved","all_questions","all_questions_weaker_ridge")}
    result = dict(post_hoc=True,scope="static final readout fit; no claim of improved closed-loop health",results=results,aggregate=aggregate)
    write(output,result)
    print(json.dumps(dict(cases=len(results),aggregate=aggregate)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("raw",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    diagnose(args.raw,args.output)
