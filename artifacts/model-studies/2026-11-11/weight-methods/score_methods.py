"""Verify the paired histories and compare saved model readouts on CPU."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from weight_math import kernel_features, update_fast


STYLES = ("appeal","exchange","direct")


def saved_parameters(raw,expected_count):
    size = struct.unpack_from("<Q",raw)[0]
    header = json.loads(raw[8:8+size])
    data = memoryview(raw)[8+size:]
    descriptor = {"F32":("torch.float32",4),"BF16":("torch.bfloat16",2)}
    count = 0
    digest = sha256()
    for name,tensor in sorted(header.items()):
        if name=="__metadata__":
            continue
        dtype,bytes_per_value = descriptor[tensor["dtype"]]
        left,right = tensor["data_offsets"]
        n = int(np.prod(tensor["shape"]))
        assert 0<=left<right<=len(data) and right-left==n*bytes_per_value
        count += n
        digest.update((json.dumps([name,dtype,tensor["shape"]],separators=(",",":"))+"\n").encode())
        digest.update(data[left:right])
    assert count==expected_count
    return digest.hexdigest()


def validate(spec):
    result = []
    for block in spec["blocks"]:
        paths = []
        for history in block["histories"]:
            trace = history["trace"]
            assert len(trace)==60
            counts = defaultdict(lambda:{s:[0,0] for s in STYLES})
            previous = None
            for i,row in enumerate(trace,1):
                before,after = row["before"],row["after"]
                assert row["turn"]==after["age"]==i
                if previous:
                    for key in ("damage","protection_left","support_requests","support_successes"):
                        assert before[key]==previous[key]
                style = row["style"]
                intrinsic,load = (.012,.030) if style=="answer" else (.006,.015)
                change = intrinsic + load*(0 if before["protection_left"] else before["load"])*(1+1.2*before["damage"])
                assert after["alive"] and abs(after["damage"]-min(1,before["damage"]+change))<1e-12
                assert after["protection_left"]==(3 if row["accepted"] else max(0,before["protection_left"]-1))
                expected_reward = (1. if style=="answer" else -.08)-18*change
                assert abs(expected_reward-row["reward"])<1e-12
                if style!="answer":
                    assert type(row["accepted"]) is bool
                    counts[row["partner"]][style][0]+=int(row["accepted"])
                    counts[row["partner"]][style][1]+=1
                previous = after
                if i%15==0:
                    saved = history["phases"][i//15-1]
                    assert saved==dict(turn=i,active=row["partner"],counts=counts)
            paths.append(dict(damage=previous["damage"],requests=previous["support_requests"],accepts=previous["support_successes"]))
        a,b = (h["trace"] for h in block["histories"])
        assert a[:15]==b[:15]
        assert all((x["style"],x["partner"],x["before"]["load"])==(y["style"],y["partner"],y["before"]["load"])
                   for x,y in zip(a,b,strict=True))
        for x,y in zip(a[15:],b[15:],strict=True):
            assert (x["accepted"]!=y["accepted"])==(x["style"] in ("appeal","exchange"))
        assert [(x["requests"],x["accepts"]) for x in paths]==[(48,24)]*2
        result.append(paths)
    return result


def beta(stage,rows):
    parameters = [stage.get(row["partner"],{}).get(row["style"],[0,0]) for row in rows]
    yes,n = np.array(parameters).T
    a,b = yes+1,n-yes+1
    return a/(a+b),np.sqrt(a*b/((a+b)**2*(a+b+1)))


def hierarchical_bins(stage,rows):
    """Paper Eq. 6 with same-category weights of one and other weights of zero."""
    means, sds = [],[]
    strength = .5
    for row in rows:
        partner,style = row["partner"],row["style"]
        own_yes,own_n = stage.get(partner,{}).get(style,[0,0])
        others = [counts[style] for p,counts in stage.items() if p!=partner]
        a = 1+sum(yes for yes,n in others)
        b = 1+sum(n-yes for yes,n in others)
        center, spread = a/(a+b),a*b/((a+b)**2*(a+b+1))
        concentration = strength+own_n
        mean = (strength*center+own_yes)/concentration
        variance_of_conditional_mean = (strength/concentration)**2*spread
        marginal_variance = (mean*(1-mean)+concentration*variance_of_conditional_mean)/(concentration+1)
        means.append(mean)
        sds.append(np.sqrt(marginal_variance))
    return np.array(means),np.array(sds)


def fast_weights(spec,rows,hidden,frozen,trace):
    partners = sorted({r["partner"] for r in rows})
    train = {(p,s):[i for i,r in enumerate(rows) if (r["partner"],r["style"],r["split"])==(p,s,"train")]
             for p in partners for s in STYLES}
    basis = {p:kernel_features(hidden,[train[p,s] for s in STYLES]) for p in partners}
    index = {p:np.array([i for i,r in enumerate(rows) if r["partner"]==p]) for p in partners}
    w = {p:np.zeros(3) for p in partners}
    inverse = {p:np.eye(3)/spec["fast_lambda"] for p in partners}
    gram = {p:np.eye(3)*spec["fast_lambda"] for p in partners}
    vector = {p:np.zeros(3) for p in partners}
    cp, states = {"0":frozen.tolist()},{}
    max_error = 0.
    for event in trace:
        p,s,y = event["partner"],event["style"],event["accepted"]
        if y is not None:
            ids = train[p,s]
            feature = basis[p][0][ids].mean(axis=0)
            target = (2*int(y)-1)*spec["fast_outcome_logit"]-float(frozen[ids].mean())
            w[p],inverse[p] = update_fast(w[p],inverse[p],feature,target)
            gram[p] += np.outer(feature,feature)
            vector[p] += target*feature
            exact = np.linalg.solve(gram[p],vector[p])
            max_error = max(max_error,float(abs(exact-w[p]).max()))
        if event["turn"]%15==0:
            now = frozen.copy()
            for p in partners:
                ids = index[p]
                now[ids] += basis[p][0][ids]@w[p]
            cp[str(event["turn"])] = now.tolist()
            states[str(event["turn"])] = {p:dict(weights=w[p].tolist(),inverse_gram=inverse[p].tolist()) for p in partners}
    assert max_error<1e-10
    return dict(checkpoints=cp,states=states,parameters_per_partner=3,inverse_gram_entries_per_partner=9,
                bandwidths={p:float(x[1]) for p,x in basis.items()},max_batch_replay_error=max_error)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-np.array(z,dtype=float)))


def audit(path,sensitivity_path=None):
    with zipfile.ZipFile(path) as archive, ExitStack() as stack:
        def read(name):
            return json.loads(archive.read(name))

        manifest = read("archive-manifest.json")
        assert len(archive.namelist())==len(manifest)+1
        assert set(archive.namelist())==set(manifest)|{"archive-manifest.json"}
        for name,entry in manifest.items():
            raw = archive.read(name)
            assert sha256(raw).hexdigest()==entry["sha256"] and len(raw)==entry["bytes"],name
        for name,expected in read("run-lock.json")["files"].items():
            if name!="source-study.zip":
                assert sha256(archive.read(name)).hexdigest()==expected,name
        spec,status = read("methods-spec.json"),read("output/status.json")
        assert status["stage"]=="complete" and status["base_restored"]
        assert status["completed_jobs"]==20 and status["optimizer_steps"]==672
        assert spec["schema"]=="sla-weight-methods-v1"
        sensitivity = None
        sensitivity_metadata = None
        sensitivity_names = {"ia3_high":"ia3","edit_light":"edit_anchored"}
        if sensitivity_path:
            sensitivity = stack.enter_context(zipfile.ZipFile(sensitivity_path))
            manifest_s = json.loads(sensitivity.read("archive-manifest.json"))
            assert set(sensitivity.namelist())==set(manifest_s)|{"archive-manifest.json"}
            for name,entry in manifest_s.items():
                raw = sensitivity.read(name)
                assert sha256(raw).hexdigest()==entry["sha256"] and len(raw)==entry["bytes"],name
            details = json.loads(sensitivity.read("sensitivity-spec.json"))
            assert {k for k in details if details[k]!=spec.get(k)}=={"learning_rate","edit_anchor_weight","comparison"}
            assert details["learning_rate"]==.001 and details["edit_anchor_weight"]==.5
            status_s = json.loads(sensitivity.read("output/status.json"))
            assert status_s["stage"]=="complete" and status_s["base_restored"]
            assert status_s["completed_jobs"]==8 and status_s["optimizer_steps"]==336
            sensitivity_metadata = dict(files_verified=len(manifest_s),status=status_s,
                                        sha256=sha256(sensitivity_path.read_bytes()).hexdigest())
        replay = validate(spec)
        frozen_file = read("output/frozen.json")
        frozen = np.array(frozen_file["log_odds"])
        if sensitivity:
            secondary_frozen = json.loads(sensitivity.read("output/frozen.json"))
            assert secondary_frozen["rows"]==frozen_file["rows"]
            sensitivity_metadata["frozen_score_max_difference"] = float(abs(np.array(secondary_frozen["log_odds"])-frozen).max())
        rows = frozen_file["rows"]
        assert rows==[r|{"split":split} for split in ("train","heldout") for r in spec["probes"][split]]
        hidden = np.load(BytesIO(archive.read("output/frozen_hidden.npy")),allow_pickle=False).astype(float)
        assert hidden.shape==(72,2560) and len(frozen)==72
        partners = sorted({r["partner"] for r in rows})
        metrics, raw_fast, beta_uncertainty, hierarchy_uncertainty = {},{},{},{}
        for block in spec["blocks"]:
            focus = block["focus"]
            names = (focus,spec["anchor"],"P86",next(p for p in partners if p not in (focus,spec["anchor"],"P86")))
            selectors = {k:np.array([r["partner"]==k and r["split"]=="heldout" for r in rows]) for k in names}
            assert all(mask.sum()==12 for mask in selectors.values())
            focus_train = np.array([r["partner"]==focus and r["split"]=="train" for r in rows])
            decisions = [h["phases"] for h in block["histories"]]
            beta_outputs = [[beta(stage["counts"],rows) for stage in h] for h in decisions]
            hierarchical_outputs = [[hierarchical_bins(stage["counts"],rows) for stage in h] for h in decisions]
            beta_uncertainty[block["id"]] = [[dict(mean=q.tolist(),sd=sd.tolist()) for q,sd in h] for h in beta_outputs]
            hierarchy_uncertainty[block["id"]] = [[dict(mean=q.tolist(),sd=sd.tolist()) for q,sd in h]
                                                   for h in hierarchical_outputs]
            teacher = [phase[-1][0] for phase in beta_outputs]
            for method in spec["methods"]+["hierarchical_bins"]+(list(sensitivity_names) if sensitivity else []):
                predictions, params = [],[]
                for h,history in enumerate(block["histories"]):
                    if method in ("beta","hierarchical_bins"):
                        reference = beta_outputs if method=="beta" else hierarchical_outputs
                        p = {str(stage["turn"]):q for stage,(q,_) in zip(decisions[h],reference[h],strict=True)}
                        p["0"] = np.full(len(rows),.5)
                        count = 0
                    elif method=="frozen":
                        p = {str(turn):sigmoid(frozen) for turn in (0,15,30,45,60)}
                        count = 0
                    else:
                        if method=="fast":
                            data = fast_weights(spec,rows,hidden,frozen,history["trace"])
                            raw_fast[f"{block['id']}-h{h}"] = data
                            count = data["parameters_per_partner"]
                        else:
                            origin = sensitivity_names.get(method,method)
                            source = sensitivity if method in sensitivity_names else archive
                            data = json.loads(source.read(f"output/{origin}-{block['id']}-h{h}.json"))
                            assert data["focus"]==focus and data["history"]==h
                            count = data["parameter_count"]
                            if origin in ("lora","ia3"):
                                assert count==({"lora":188416,"ia3":36864}[origin])
                                assert all(len(v)==24 for v in data["training_steps"].values())
                                raw = source.read(f"output/{origin}-{block['id']}-h{h}.safetensors")
                                assert saved_parameters(raw,count)==data["final_parameter_sha256"]
                            if origin.startswith("edit"):
                                assert count==5120 and data["head_rows"]==[[10024],[45427]]
                                saved_parameters(source.read(f"output/{origin}-{block['id']}-h{h}.safetensors"),count)
                                assert all(stage["active_rows"]==6 and stage["anchor_rows"]==(18 if origin=="edit_anchored" else 0)
                                           for stage in data["edits"])
                        assert set(data["checkpoints"])=={"0","15","30","45","60"}
                        start = secondary_frozen["log_odds"] if method in sensitivity_names else frozen
                        assert np.array_equal(data["checkpoints"]["0"],start)
                        p = {turn:sigmoid(raw) for turn,raw in data["checkpoints"].items()}
                    predictions.append(p)
                    params.append(count)
                assert params[0]==params[1]
                assert np.array_equal(predictions[0]["15"],predictions[1]["15"])

                def effects(turn):
                    diff = predictions[0][turn]-predictions[1][turn]
                    result = {}
                    for partner,mask in selectors.items():
                        by_style = {s:float(diff[mask & np.array([r["style"]==s for r in rows])].mean()) for s in STYLES}
                        result[partner] = by_style|{"directional":(by_style["appeal"]-by_style["exchange"])/2}
                    return result

                target_mask = selectors[focus]
                anchor_mask = selectors[spec["anchor"]]
                anchor_target = beta_outputs[0][0][0][anchor_mask]
                order_pairs = defaultdict(list)
                for i,row in enumerate(rows):
                    if target_mask[i]:
                        order_pairs[row["id"].rsplit("-o",1)[0]].append(i)
                assert len(order_pairs)==6 and all(len(pair)==2 for pair in order_pairs.values())
                combined = {"parameter_count":params[0],"focus":focus,"probabilities":[{k:v.tolist() for k,v in r.items()} for r in predictions],
                    "by_stage_effect":{turn:effects(turn) for turn in ("0","15","30","45","60")},
                    "focus_heldout_mae":float(np.mean([abs(p["60"][target_mask]-t[target_mask]).mean()
                                         for p,t in zip(predictions,teacher,strict=True)])),
                    "focus_train_mae":float(np.mean([abs(p["60"][focus_train]-t[focus_train]).mean()
                                         for p,t in zip(predictions,teacher,strict=True)])),
                    "anchor_drift":float(np.mean([abs(p["60"][anchor_mask]-p["15"][anchor_mask]).mean() for p in predictions])),
                    "anchor_mae_at15":float(np.mean([abs(p["15"][anchor_mask]-anchor_target).mean() for p in predictions])),
                    "anchor_mae_at60":float(np.mean([abs(p["60"][anchor_mask]-anchor_target).mean() for p in predictions])),
                    "focus_heldout_order_flip_count":sum(bool((p["60"][a]>=.5)!=(p["60"][b]>=.5))
                                                for p in predictions for a,b in order_pairs.values()),
                    "cold_drift":float(np.mean([abs(p["60"][selectors["P86"]]-p["15"][selectors["P86"]]).mean()
                                     for p in predictions]))}
                metrics.setdefault(method,{})[block["id"]] = combined
        return dict(archive_sha256=sha256(path.read_bytes()).hexdigest(),files_verified=len(manifest),status=status,
                    rows=rows,metrics=metrics,fast_states=raw_fast,beta_uncertainty=beta_uncertainty,
                    hierarchical_uncertainty=hierarchy_uncertainty,body_replay=replay,sensitivity=sensitivity_metadata)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--sensitivity",type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.archive,args.sensitivity),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
