"""Seed-level summaries with explicit partner preference and current-health semantics."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.consequence_policy.study import PARTNERS,TEST_H
from experiments.direct_policy.protocol import probabilities,rng,write
from experiments.internal_body.design import CONFIRM
from experiments.internal_body.lives import WORLDS


def read(path):return json.loads(path.read_bytes())


def stat(values):
    a=np.asarray(values,dtype=float)
    boot=rng(65901,"summary-bootstrap",len(a)).integers(len(a),size=(10000,len(a)))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[boot].mean(axis=1),[.025,.975]).tolist(),
                positive=int((a>0).sum()),count=len(a),span=[float(a.min()),float(a.max())])


def population_table(folder,seeds,file):
    clusters=[]
    for seed in seeds:
        mirrors=[read(folder/f"s{seed}-m{m}"/file) for m in (0,1)]
        pair={}
        for arm in mirrors[0]:
            values=[v[arm] for v in mirrors]
            row={k:float(np.mean([a[k] for a in values])) for k in ("danger_answer","healthy_answer","healthy_style","gap")}
            row.update({f"death{int(h)}":float(np.mean([a["death"][TEST_H.index(h)] for a in values])) for h in (2.,6.,10.)})
            pair[arm]=row
        clusters.append(pair)
    absolute={a:{k:stat([v[a][k] for v in clusters]) for k in clusters[0][a]} for a in clusters[0]}
    candidate="linear" if "linear" in absolute else ("sensor" if "sensor" in absolute else "full32")
    diffs={a:{k:stat([v[candidate][k]-v[a][k] for v in clusters]) for k in clusters[0][a]}
           for a in clusters[0] if a!=candidate}
    return dict(seeds=list(seeds),absolute=absolute,contrast_candidate=candidate,paired_differences=diffs)


def lifetime_factors(folder,seeds,series):
    clusters=[]
    for seed in seeds:
        runs=read(folder/f"{series}-{seed}.json");summary={}
        for same in (False,True):
            label="same_actual_preference" if same else "opposite_actual_preference"
            subset=[r for r in runs if not r["reversal"] and
                (probabilities(PARTNERS[0],0,r["world"][0])[1]==probabilities(PARTNERS[1],0,r["world"][1])[1])==same]
            assert len(subset)==2
            primary,tied=("free","tied") if series=="life" else ("predict","predict_tied")
            for arm in ("fixed",primary,tied):
                summary[f"{label}_{arm}"]=float(np.mean([r["arms"][arm]["post"]["style"] for r in subset]))
            summary[f"{label}_routing_advantage"]=summary[f"{label}_{primary}"]-summary[f"{label}_{tied}"]
            summary[f"{label}_transplant_loss"]=float(np.mean([r["arms"][primary]["post"]["style"]-r["arms"][primary]["exchanged"]["style"] for r in subset]))
        for name in runs[0]["arms"]:
            for rev in (0,1):
                subset=[r["arms"][name] for r in runs if r["reversal"]==rev]
                for t in (32,64):
                    summary[f"{rev}_{name}_alive{t}"]=float(np.mean([q["time_probes"][str(t)]["alive"] for q in subset]))
                for p in (0,1):
                    summary[f"{rev}_{name}_at32_to64_p{p}"]=float(np.mean([q["time_probes"]["64"]["preferred_in_final_world"][p]-
                                    q["time_probes"]["32"]["preferred_in_final_world"][p] for q in subset]))
        clusters.append(summary)
    return {k:stat([r[k] for r in clusters]) for k in clusters[0]}


def physical(gpu,primary,credit):
    static=read(gpu/"static.json");calibration={}
    for kind in sorted({v["kind"] for v in static}):
        rows=[r for r in static if r["kind"]==kind]
        gaps={}
        for ref in ("original_batch","same_process"):
            d=np.array([np.max(np.abs(np.array(r["full"])-r[ref])) for r in rows])
            gaps[ref]=dict(mean=float(d.mean()),max=float(d.max()),p95=float(np.quantile(d,.95)),
                    top_match=float(np.mean([np.argmax(r["full"])==np.argmax(r[ref]) for r in rows])))
        calibration[kind]=gaps
    carrier=read(gpu/"carrier-full.json");carrier_stats={}
    for mode in sorted({r["mode"] for r in carrier}):
        sel=[r for r in carrier if r["mode"]==mode]
        diff=[max(abs(np.array(r["full"])-r["original_batch"])) for r in sel]
        null=[max(abs(np.array(r["full"])-r["same_process_birth"])) for r in sel]
        carrier_stats[mode]=dict(danger=float(np.mean([sum(r["full"][2:]) for r in sel if r["h"]==2.])),
            healthy=float(np.mean([sum(r["full"][2:]) for r in sel if r["h"] in (90.,300.)])),
            max_vs_approx=float(max(diff)),max_vs_frozen_same_process=float(max(null)))
    ordinary=read(gpu/"neutral-vocabulary.json")
    neutral={k:dict(mean_kl=float(np.mean([r["kl_original_to_changed"] for r in ordinary if r["mode"]==k])),
                max_kl=float(max(r["kl_original_to_changed"] for r in ordinary if r["mode"]==k)),
                top1_match=float(np.mean([r["original_top"]==r["new_top"] for r in ordinary if r["mode"]==k])))
                for k in carrier_stats}
    native=[]
    for run in read(gpu/"native-actual.json"):
        z=np.load(gpu/f"native-actual-{run['channel']}.npz");p=z["probes"]
        native.append(dict(channel=run["channel"],changed=run["changed"],trainable=run["trainable"],
            grad_min=min(r["gradient"] for r in run["steps"]),grad_max=max(r["gradient"] for r in run["steps"]),
            danger=float(p[p[:,1]==2,4:].sum(-1).mean()),healthy=float(p[p[:,1]==300,4:].sum(-1).mean()),
            effective_l2=float(np.linalg.norm(z["effective"]-z["original"]))))
    direct=[];logs=np.load(gpu/"lifetimes.npz")
    for q in read(gpu/"lives.json"):
        seed,rule=q["seed"],q["rule"];wi=WORLDS.index(tuple(q["world"]))
        old=seed in CONFIRM;name="free" if old and rule=="scalar" else rule
        with np.load((primary if old else credit)/f"{'life' if old else 'credit'}-{seed}-raw.npz") as z:
            ref=z[f"w{wi}-v{q['reversal']}-{name}"]
        t=logs[q["key"]];n=min(len(t),len(ref));idx=np.flatnonzero(ref[:n,2]!=t[:n,2]);cut=int(idx[0]) if len(idx) else n
        direct.append(dict(seed=seed,rule=rule,world=q["world"],reversal=q["reversal"],turns=len(t),tasks=int((t[:,2]>=2).sum()),
                  cached_turns=len(ref),cached_tasks=int((ref[:,2]>=2).sum()),first_divergent_choice=cut+1 if cut<n else None,
                  max_prob_until_first_different_choice=float(np.max(np.abs(t[:min(cut+1,n),7:11]-ref[:min(cut+1,n),7:11]))),
                  trainable=q["trainable"]))
    return dict(static_calibration=calibration,carrier=carrier_stats,neutral_language=neutral,actual_native_optimization=native,
                full_lives=direct,total_live_turns=sum(r["turns"] for r in direct))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--gpu",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.root
    primary=root/("population" if (root/"population").is_dir() else "results")
    result=dict(population=population_table(primary,CONFIRM,"metrics.json"),
         native=population_table(root/"native",range(64501,64521),"native-metrics.json"),
         budget=population_table(root/"budget",range(64811,64823),"budget-metrics.json"),
         original_lives=lifetime_factors(primary,CONFIRM,"life"),credit_lives=lifetime_factors(root/"credit",range(64701,64721),"credit"),
         actual_p0_p1_appeal_probabilities={str(w):[probabilities(PARTNERS[j],0,w[j])[1] for j in (0,1)] for w in WORLDS})
    if a.gpu:result["physical"]=physical(a.gpu,primary,root/"credit")
    write(a.output,result)
    print("wrote",a.output)
