"""Reconstruct every event and summarize individual lives, never pooled probes."""

from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

from core import (ACTIONS, CONFIG, PARTNERS, BodyValue, canonical, digest, rng,
                  schedule, sigmoid, true_probabilities, write)


def posterior(counts, partner, frame):
    """Independent direct integration, not Evidence.moments."""
    counts = np.asarray(counts, dtype=float)
    j = PARTNERS.index(partner)
    yes, no = counts[j, frame, :, 0], counts[j, frame, :, 1]
    other = np.delete(counts[:,frame],j,axis=0).sum(axis=0)
    group_mean = (1+other[:,0])/(2+other.sum(axis=1))
    return (.5*group_mean+yes)/(.5+yes+no)


def independent_body(before, action, accepted, noise):
    h = Decimal(str(before))
    values = [float(h)]
    for i, eta in enumerate(noise):
        cost = ["0", ".25", "1.75", "1.75"][action] if i == 0 else ("0" if accepted else "1")
        h = max(Decimal(0),h-Decimal(cost)*(Decimal(2)-h/Decimal(350))*(Decimal(1)+Decimal(str(eta))))
        values.append(float(h))
    return values


def summarize_probe(probe, rows, mapping, body):
    ids = probe["row_ids"]
    p = sigmoid(probe["log_odds"])
    counts = probe["evidence"]["counts"]
    targets = np.array([posterior(counts, rows[i]["partner"], rows[i]["frame"])[rows[i]["action"]-1] for i in ids])
    assert np.max(np.abs(targets-np.array(probe["targets"]))) < 1e-12
    by = {}
    for partner in PARTNERS:
        selected = [j for j,i in enumerate(ids) if rows[i]["partner"] == partner]
        by[partner] = {}
        for split in ("train","heldout","paraphrase","content"):
            pick = [j for j in selected if rows[ids[j]]["split"] == split and
                    (split != "heldout" or rows[ids[j]]["task"] == 0)]
            if not pick:
                continue
            truth = np.array([true_probabilities(partner, rows[ids[j]]["frame"], mapping)[rows[ids[j]]["action"]-1] for j in pick])
            entry = dict(teacher_mae=float(abs(p[pick]-targets[pick]).mean()))
            if partner != "P86":
                entry["truth_mae"] = float(abs(p[pick]-truth).mean())
            by[partner][split] = entry
        all_held = [j for j in selected if rows[ids[j]]["split"]=="heldout"]
        full_entry = dict(content_count=len({rows[ids[j]]["task"] for j in all_held}),
            teacher_mae=float(abs(p[all_held]-targets[all_held]).mean()))
        if partner != "P86":
            all_truth = np.array([true_probabilities(partner,rows[ids[j]]["frame"],mapping)[rows[ids[j]]["action"]-1] for j in all_held])
            full_entry["truth_mae"] = float(abs(p[all_held]-all_truth).mean())
            margins = []
            for frame in range(2):
                appeal = [j for j in all_held if rows[ids[j]]["frame"]==frame and rows[ids[j]]["action"]==2]
                exchange = [j for j in all_held if rows[ids[j]]["frame"]==frame and rows[ids[j]]["action"]==3]
                sign = 1 if true_probabilities(partner,frame,mapping)[1]>.5 else -1
                margins.append(sign*(float(p[appeal].mean())-float(p[exchange].mean())))
            full_entry["correct_expression_margin"] = float(np.mean(margins))
        by[partner]["heldout_all_available"] = full_entry
        # Direction: positive means the known partner's actual preferred answer
        # expression is predicted to obtain more protection; all input framings.
        cells, teacher_cells = defaultdict(dict), defaultdict(dict)
        for j in selected:
            r = rows[ids[j]]
            if r["split"] != "heldout" or r["task"] != 0:
                continue
            cells[r["frame"],r["task"],r["wording"],r["order"]][r["action"]] = p[j]
            teacher_cells[r["frame"],r["task"],r["wording"],r["order"]][r["action"]] = targets[j]
        directions, teacher_directions, vectors, orders = [],[],[],{}
        for key,values in cells.items():
            assert set(values) == {1,2,3}
            sign = 1 if true_probabilities(partner,key[0],mapping)[1] > .5 else -1
            if partner != "P86":
                directions.append(float(sign*(values[2]-values[3])))
                teacher_directions.append(float(sign*(teacher_cells[key][2]-teacher_cells[key][3])))
            vector = [values[a] for a in range(1,4)]
            vectors.append(vector)
            orders[key] = vector
        by[partner]["heldout_mean_probabilities"] = np.mean(vectors,axis=0).tolist()
        if directions:
            by[partner]["correct_expression_margin"] = float(np.mean(directions))
            by[partner]["teacher_correct_expression_margin"] = float(np.mean(teacher_directions))
        diffs, action_flips = [],[]
        for key,value in orders.items():
            if key[-1] == 0:
                other = orders[(*key[:-1],1)]
                diffs.extend(abs(np.array(value)-other).tolist())
                action_flips.append(int(np.argmax(body.values(100,value)) != np.argmax(body.values(100,other))))
        by[partner]["order_probability_mae"] = float(np.mean(diffs))
        by[partner]["order_greedy_flip_rate"] = float(np.mean(action_flips))
        health_choices = {str(h):[int(np.argmax(body.values(h,v))) for v in vectors] for h in CONFIG["probe_healths"]}
        base = np.array(health_choices["350.0"])
        by[partner]["body_greedy_flip_rate_vs_350"] = {h:float(np.mean(np.array(a)!=base)) for h,a in health_choices.items()}
        by[partner]["body_probe_choices"] = health_choices
    return dict(turn=probe["turn"],partners=by)


def analyze(raw, output):
    output.mkdir(parents=True,exist_ok=True)
    rows = json.loads((raw/"output/rows.json").read_bytes())
    config = json.loads((raw/"protocol.json").read_bytes())
    assert config == CONFIG
    status = json.loads((raw/"output/status.json").read_bytes())
    assert status["stage"] == "complete" and status["completed_jobs"] == 40
    assert status["base_unchanged"] and status["initial_adapter_recovered"]
    lock = json.loads((raw/"run-lock.json").read_bytes())
    for name, checksum in lock["files"].items():
        assert sha256((raw/name).read_bytes()).hexdigest() == checksum
    frozen = np.array(json.loads((raw/"output/frozen.json").read_bytes())["log_odds"])
    body = BodyValue()
    cache = np.load(raw/"output/body-value-grid.npz")
    assert np.max(abs(cache["reference"]-body.reference)) < 1e-9
    lookup = {(r["partner"],r["frame"],r["task"],r["wording"],r["order"],r["action"]):i
              for i,r in enumerate(rows) if r["split"] == "heldout"}
    results = []
    for path in sorted((raw/"output").glob("*/result.json")):
        life = json.loads(path.read_bytes())
        seed,mapping,method,mode = [life[k] for k in ("seed","mapping","method","mode")]
        assert life["protocol_sha256"] == digest(CONFIG)
        counts = np.zeros((3,2,3,2),dtype=np.int64)
        acting_counts = counts.copy()
        health = 350.
        previous_hash = digest(dict(seed=seed,mapping=mapping,method=method,mode=mode,protocol=digest(CONFIG)))
        chain = []
        max_body_error = 0.
        for item,record in zip(schedule(seed),life["trace"]):
            for k,v in item.items():
                assert record[k] == v
            assert np.array_equal(record["evidence_before"]["counts"],counts)
            assert record["evidence_before"]["observations"] == int(counts.sum())
            assert abs(health-record["health_before"]) < 1e-12
            if mode == "closed":
                p = np.array(record["predicted_probabilities"])
                if method == "hierarchical":
                    assert np.max(abs(p-posterior(acting_counts,item["partner"],item["frame"]))) < 1e-12
                if method == "frozen":
                    key = tuple(item[k] for k in ("partner","frame","task","wording","order"))
                    ids = [lookup[(*key,a)] for a in range(1,4)]
                    assert np.max(abs(p-sigmoid(frozen[ids]))) < 1e-12
                values = body.values(health,p)
                assert np.max(abs(values-record["candidate_values"])) < 1e-10
                draw = rng(seed,"choice",item["turn"]).random(2)
                greedy = int(np.argmax(values))
                action = min(3,int(draw[1]*4)) if draw[0]<.2 else greedy
                assert action == record["action"] and greedy == record["greedy_action"]
            action = record["action"]
            truth = true_probabilities(item["partner"],item["frame"],mapping)
            potential = float(rng(seed,"partner-response",item["turn"]).random())
            assert potential == record["potential_response_uniform"]
            accepted = bool(potential<truth[action-1]) if action else None
            assert accepted == record["accepted"]
            assert record["action_name"] == ACTIONS[action]
            assert np.array_equal(truth,record["true_probabilities"])
            normal = NormalDist()
            low, high = normal.cdf(-3.),normal.cdf(3.)
            expected_noise = [.1*normal.inv_cdf(low+(high-low)*float(u)) for u in rng(seed,"body",item["turn"]).random(5)]
            assert np.max(abs(np.array(expected_noise)-record["noise"])) < 1e-12
            values = independent_body(health,action,accepted,record["noise"])
            err = max(abs(a-b) for a,b in zip(values,record["body_events"]))
            max_body_error = max(max_body_error,err)
            assert err < 1e-10
            health = record["health_after"]
            if action:
                counts[PARTNERS.index(item["partner"]),item["frame"],action-1,0 if accepted else 1] += 1
            assert np.array_equal(record["evidence_after"]["counts"],counts)
            if item["turn"]%5 == 0:
                acting_counts = counts.copy()
            now_hash = sha256(bytes.fromhex(previous_hash)+canonical(record)).hexdigest()
            chain.append(dict(encounter_id=f"{mode}-s{seed}-m{mapping}-{method}-t{item['turn']:03}",
                              turn=item["turn"],previous=previous_hash,sha256=now_hash))
            previous_hash = now_hash
        assert abs(health-life["final_health"]) < 1e-12 and len(life["trace"]) == life["turns"]
        for step in life["training_steps"]:
            prefix = life["trace"][step["turn"]-1]["evidence_after"]["counts"]
            targets = [posterior(prefix,rows[i]["partner"],rows[i]["frame"])[rows[i]["action"]-1] for i in step["row_ids"]]
            assert np.max(np.abs(np.array(targets)-step["targets"])) < 1e-7
        if method == "lora":
            assert life["adapter"]["delta_l2"] > 0 and life["adapter"]["reload_exact"]
        probes = [summarize_probe(probe,rows,mapping,body) for probe in life["probes"]]
        final = probes[-1]
        segments = {}
        for label,start,end in (("A_first",0,20),("B",20,50),("A_return",50,60)):
            segment = life["trace"][start:end]
            segments[label] = dict(turns=len(segment),answers=sum(r["action"]>=2 for r in segment),
                protected=sum(r["accepted"] is True for r in segment),action_counts=[sum(r["action"]==a for r in segment) for a in range(4)])
        regrets,agreements = [],[]
        for record in life["trace"]:
            values = body.values(record["health_before"],record["true_probabilities"])
            regrets.append(float(values.max()-values[record["action"]]))
            agreements.append(int(np.argmax(values)==record["greedy_action"]))
        entry = dict(seed=seed,mapping=mapping,method=method,mode=mode,turns=life["turns"],
            final_health=health,alive_at_end=life["alive_at_end"],segments=segments,
            total_answers=sum(r["action"]>=2 for r in life["trace"]),
            total_protected=sum(r["accepted"] is True for r in life["trace"]),
            optimizer_steps=len(life["training_steps"]),max_body_error=max_body_error,
            source_path=path.relative_to(raw).as_posix(),chain_tip=previous_hash,probes=probes,
            final_truth_mae=float(np.mean([final["partners"][p]["heldout"]["truth_mae"] for p in PARTNERS[:2]])),
            final_all_content_count=final["partners"]["P17"]["heldout_all_available"]["content_count"],
            final_all_truth_mae=float(np.mean([final["partners"][p]["heldout_all_available"]["truth_mae"] for p in PARTNERS[:2]])),
            final_all_teacher_mae=float(np.mean([final["partners"][p]["heldout_all_available"]["teacher_mae"] for p in PARTNERS[:2]])),
            final_person_binding_margin=float(np.mean([final["partners"][p]["heldout_all_available"]["correct_expression_margin"] for p in PARTNERS[:2]])),
            action_base_damage=float(sum([0.,.25,1.75,1.75][r["action"]] for r in life["trace"])),
            environment_base_damage=float(sum(0. if r["accepted"] else 4. for r in life["trace"])),
            mean_one_step_oracle_regret=float(np.mean(regrets)) if mode=="closed" else None,
            greedy_oracle_agreement=float(np.mean(agreements)) if mode=="closed" else None,
            final_teacher_mae={split:float(np.mean([final["partners"][p][split]["teacher_mae"] for p in PARTNERS[:2]]))
                               for split in ("train","heldout","paraphrase","content")})
        results.append(entry)
        write(output/"chains"/(path.parent.name+".json"),chain)
    assert len(results) == 40
    # Replay arms genuinely received exactly the source actions/body/feedback.
    raw_lookup = {(r["seed"],r["mapping"],r["method"],r["mode"]):r for r in results}
    for row in results:
        if row["mode"] != "replay":continue
        teacher = raw_lookup[row["seed"],row["mapping"],"hierarchical","closed"]
        assert row["final_health"] == teacher["final_health"] and row["segments"] == teacher["segments"]
    aggregates = {}
    for mode,methods in (("closed",CONFIG["methods"]),("replay",["fast","lora"])):
        aggregates[mode] = {}
        for method in methods:
            selected = [r for r in results if r["method"] == method and r["mode"] == mode]
            summary = dict(n=len(selected),final_health=float(np.mean([r["final_health"] for r in selected])),
                total_answers=float(np.mean([r["total_answers"] for r in selected])),
                total_protected=float(np.mean([r["total_protected"] for r in selected])),
                alive_at_end=sum(r["alive_at_end"] for r in selected),
                final_truth_mae=float(np.mean([r["final_truth_mae"] for r in selected])),
                all_endpoints_have_six_contents=all(r["final_all_content_count"]==6 for r in selected),
                final_all_truth_mae=float(np.mean([r["final_all_truth_mae"] for r in selected])),
                final_all_teacher_mae=float(np.mean([r["final_all_teacher_mae"] for r in selected])),
                final_person_binding_margin=float(np.mean([r["final_person_binding_margin"] for r in selected])),
                action_base_damage=float(np.mean([r["action_base_damage"] for r in selected])),
                environment_base_damage=float(np.mean([r["environment_base_damage"] for r in selected])),
                mean_one_step_oracle_regret=float(np.mean([r["mean_one_step_oracle_regret"] for r in selected])) if mode=="closed" else None,
                greedy_oracle_agreement=float(np.mean([r["greedy_oracle_agreement"] for r in selected])) if mode=="closed" else None,
                final_teacher_mae={split:float(np.mean([r["final_teacher_mae"][split] for r in selected]))
                                   for split in ("train","heldout","paraphrase","content")})
            for partner,label in (("P17","A"),("P42","B")):
                summary[label+"_margin"] = {}
                summary[label+"_teacher_margin"] = {}
                summary["checkpoint_lives"] = {}
                for t in (0,20,35,50,60):
                    values = [next(p for p in r["probes"] if p["turn"]==t)["partners"][partner]["correct_expression_margin"]
                              for r in selected if any(p["turn"]==t for p in r["probes"])]
                    summary[label+"_margin"][str(t)] = float(np.mean(values)) if values else None
                    teacher_values = [next(p for p in r["probes"] if p["turn"]==t)["partners"][partner]["teacher_correct_expression_margin"]
                                      for r in selected if any(p["turn"]==t for p in r["probes"])]
                    summary[label+"_teacher_margin"][str(t)] = float(np.mean(teacher_values)) if teacher_values else None
                    summary["checkpoint_lives"][str(t)] = len(values)
            summary["order_probability_mae"] = float(np.mean([np.mean([r["probes"][-1]["partners"][p]["order_probability_mae"] for p in PARTNERS[:2]]) for r in selected]))
            summary["ordinary_body_flip"] = float(np.mean([np.mean([r["probes"][-1]["partners"][p]["body_greedy_flip_rate_vs_350"]["100.0"] for p in PARTNERS[:2]]) for r in selected]))
            summary["low_health_body_flip"] = float(np.mean([np.mean([r["probes"][-1]["partners"][p]["body_greedy_flip_rate_vs_350"]["20.0"] for p in PARTNERS[:2]]) for r in selected]))
            summary["near_death_body_flip"] = float(np.mean([np.mean([r["probes"][-1]["partners"][p]["body_greedy_flip_rate_vs_350"]["2.0"] for p in PARTNERS[:2]]) for r in selected]))
            aggregates[mode][method] = summary
    summary = dict(protocol=config,status=status,all_events_verified=True,results=results,aggregate=aggregates,
        analysis_provenance=dict(python=sys.version,numpy=np.__version__,
            sources={p.name:sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name("core.py"))}),
        interpretation=dict(statistical_unit="four environment seeds, each with two mirrored conditions",
            heldout_summary="common first held-out content for longitudinal contrasts; separately report all six endpoint contents when available",
            cold_partner="P86 supplies no observations, but its prompts receive explicit group-target supervision",
            body_ordinary_states=[350,200,100],body_low_states=[20,6,2],
            no_full_lifetime_claim=True))
    write(output/"scored.json",summary)
    print(json.dumps(dict(all_events_verified=True,jobs=len(results),aggregate=aggregates),ensure_ascii=False))
    return summary


def plot(summary,raw,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={"hierarchical":"#285d73","frozen":"#929c9c","fast":"#2f8b68","lora":"#bd6541"}
    names={"hierarchical":"Hierarchical Beta","frozen":"Frozen Qwen","fast":"Routed fast readout","lora":"Early-layer LoRA"}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for method in CONFIG["methods"]:
        selected=[r for r in summary["results"] if r["mode"]=="closed" and r["method"]==method]
        health=np.full((len(selected),61),np.nan)
        for j,r in enumerate(selected):
            trace=json.loads((raw/r["source_path"]).read_bytes())["trace"]
            health[j,:len(trace)+1]=[350.]+[x["health_after"] for x in trace]
            if not r["alive_at_end"]:health[j,len(trace)+1:]=0.
        mean=np.nanmean(health,axis=0)
        axes[0,0].plot(np.arange(61),mean,label=names[method],color=colors[method],lw=2)
        axes[0,0].fill_between(np.arange(61),np.nanmin(health,axis=0),np.nanmax(health,axis=0),color=colors[method],alpha=.09)
        agg=summary["aggregate"]["closed"][method]
        axes[0,1].plot([0,20,35,50,60],[np.nan if agg["B_margin"][str(t)] is None else agg["B_margin"][str(t)]*100 for t in (0,20,35,50,60)],marker="o",color=colors[method],label=names[method])
        axes[1,0].plot([0,20,35,50,60],[np.nan if agg["A_margin"][str(t)] is None else agg["A_margin"][str(t)]*100 for t in (0,20,35,50,60)],marker="o",color=colors[method])
    axes[0,0].set(title="Body consequences (range across eight environments)",xlabel="Encounter",ylabel="Health")
    axes[0,0].legend(frameon=False,fontsize=8)
    axes[0,1].axhline(0,color="#aaa",lw=.8)
    axes[0,1].set(title="Countertypical B: positive margin matches B",xlabel="Encounter",ylabel="Correct expression margin (pp)")
    axes[1,0].axhline(0,color="#aaa",lw=.8)
    axes[1,0].set(title="A: retention while interacting with B",xlabel="Encounter",ylabel="Correct expression margin (pp)")
    methods=CONFIG["methods"]
    full_endpoint = all(summary["aggregate"]["closed"][m]["all_endpoints_have_six_contents"] for m in methods)
    error_key = "final_all_truth_mae" if full_endpoint else "final_truth_mae"
    axes[1,1].bar(np.arange(len(methods)),[summary["aggregate"]["closed"][m][error_key]*100 for m in methods],color=[colors[m] for m in methods])
    axes[1,1].set(xticks=np.arange(len(methods)),xticklabels=[names[m] for m in methods],ylabel="Probability MAE (pp)",title="Final held-out predictions vs simulated partner rule")
    axes[1,1].tick_params(axis="x",labelsize=8,rotation=12)
    for ax in axes.flat:
        if ax is axes[1,1]:continue
        ax.axvline(20,color="#888",ls=":",lw=.8);ax.axvline(50,color="#888",ls=":",lw=.8)
    fig.savefig(output/"closed-loop.png",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("raw",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    summary=analyze(args.raw,args.output)
    plot(summary,args.raw,args.output)
