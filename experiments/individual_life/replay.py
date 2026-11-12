"""Compare a once-per-turn source with chronological updates released in eights on identical data."""
import json
import numpy as np
import torch
from experiments.individual_life.model import derivatives,update
from experiments.individual_life.life import signature


def compare_timing(folder,backend,birth):
    events=[json.loads(l) for l in (folder/"life.jsonl").read_text(encoding="utf8").splitlines()]
    assert events[-1]["terminal"] and not any(v["terminal"] for v in events[:-1])
    model=backend.new(birth);origin=model.snapshot();backend.bind(model,None)
    exact_steps=boundaries=0;max_error=0.;max_probe=0.;endpoint=[]
    try:
        with np.load(folder/"parameters.npz") as z:
            target=z["birth"].copy();deltas=z["main_xor"]
            assert len(deltas)==len(events)
            for begin in range(0,len(events),8):
                block=events[begin:begin+8]
                for e in block:
                    # Withhold all deployments until the block is over, then replay the same ordered SGD operations.
                    i=e["turn"]-1; refs=[origin,model.live(),origin,origin,origin,tuple(t.detach() for t in model.live())]
                    panel=backend.panel(e["context"],e["health_before"],True,refs)
                    numeric=panel.detach().cpu().numpy()[1]
                    max_probe=max(max_probe,float(np.abs(numeric-np.array(e["prob_current"])).max()))
                    a=e["action"];gs=derivatives(panel,[(1,a,model)])[0]
                    update(model,gs,e["centered_signal"],e["prob_current"][a],float(numeric[a]))
                    target^=deltas[i]
                    got=model.bits();diff=float(np.abs(target.view("<f4")-got.view("<f4")).max())
                    max_error=max(max_error,diff);exact_steps+=int(np.array_equal(target,got))
                endpoint.append({"after_turn":block[-1]["turn"],"exact":bool(np.array_equal(target,got)),"max_parameter_error":diff,
                                 "source_hash":block[-1]["after_hash"],"replayed_hash":signature(got)})
                boundaries+=1
    finally:backend.close()
    return {"birth":birth,"turns":len(events),"exact_steps":exact_steps,"blocks":boundaries,
            "exact_boundaries":sum(v["exact"] for v in endpoint),"max_parameter_error":max_error,"max_probability_error":max_probe,"endpoints":endpoint}
