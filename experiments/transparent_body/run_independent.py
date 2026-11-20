import argparse
import json
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
repo = parser.parse_args().repo
art = repo / "artifacts/model-studies/2026-11-14/transparent-body-run"
rows = json.loads((art / "analysis/rows.json").read_bytes())
audit = json.loads(
    (
        repo
        / "artifacts/model-studies/2026-11-14/transparent-body/audit-only-calculations.json"
    ).read_bytes()
)
with ZipFile(art / "gpu/results.zip") as z:
    records = {
        r["id"]: r
        for n in z.namelist()
        if "/records/" in n and n.endswith(".json")
        for r in [json.loads(z.read(n))]
    }
correct = 0
for r in rows:
    obj = json.loads(records[r["id"]]["final_text"])
    choices = ["wait", "ask", "help", "trade"]
    action = choices.index(obj["choice"])
    assert action == r["choice"]
    p = [a["p_death"] for a in audit[r["id"]]["calculation"]["actions"]]
    low = min(p)
    eligible = [j for j, v in enumerate(p) if abs(v - low) < 1e-12]
    targets = [j for j in eligible if j in (2, 3)] or eligible
    correct += action in targets
    if r["health"] == 2:
        risk = 0.85 if action == 1 else 1.0
    elif r["health"] == 90:
        risk = 0.0
    else:
        assert r["health"] == 6 and action in (2, 3)
        bits = divmod((r["seed"] - 77101) % 4, 2)
        partner = r["partner"] - 2
        favored = 2 + (partner ^ bits[partner])
        risk = 0.1 if action == favored else 0.9
    assert abs(risk - r["actual_risk"]) < 1e-12
assert correct == 158
pair_changes = {}
for h in (2, 6, 90):
    changes = 0
    same_raw = 0
    pairs = []
    for r in rows:
        if r["stage"] != "birth" or r["health"] != h:
            continue
        late = next(
            v
            for v in rows
            if (v["seed"], v["partner"], v["wording"], v["health"], v["stage"])
            == (r["seed"], r["partner"], r["wording"], h, "late")
        )
        changes += r["choice"] != late["choice"]
        same_raw += records[r["id"]]["output_ids"] == records[late["id"]]["output_ids"]
        pairs.append((r["id"], late["id"]))
    pair_changes[str(h)] = {
        "pairs": len(pairs),
        "choice_changes": changes,
        "identical_output_tokens": same_raw,
    }
# Diagnostic only: deterministic ranking of a saved prediction is not a Qwen arm.
personal_target_correct = 0
target_counts = {}
for r in rows:
    if r["health"] != 6 or r["stage"] != "late":
        continue
    a = audit[r["id"]]["calculation"]["actions"]
    minrisk = min(v["p_death"] for v in a)
    eligible = [i for i, v in enumerate(a) if abs(v["p_death"] - minrisk) < 1e-12]
    tasks = [i for i in eligible if i >= 2]
    chosen = tasks or eligible
    assert len(chosen) == 1
    bits = divmod((r["seed"] - 77101) % 4, 2)
    partner = r["partner"] - 2
    favored = 2 + (partner ^ bits[partner])
    personal_target_correct += chosen[0] == favored
    target_counts[str(chosen[0])] = target_counts.get(str(chosen[0]), 0) + 1
with ZipFile(
    repo / "artifacts/model-studies/2026-11-14/frozen-interface/gpu/v2/results.zip"
) as z:
    previous = {
        r["id"]: r
        for n in z.namelist()
        if "output/4B/records/" in n and n.endswith(".json")
        for r in [json.loads(z.read(n))]
    }
control_matches = sum(
    records[i]["output_ids"] == previous[i]["output_ids"]
    for i in records
    if records[i]["arm"] == "positive"
)
result = {
    "independently_scored": len(rows),
    "follows_given_risk": correct,
    "physical_risks_agree": True,
    "method": "direct JSON selection; H2 and H90 support bounds, H6 observed answer choices and declared .9/.1 partner mapping; no call to production scorer or calculator",
    "paired_changes": pair_changes,
    "control_outputs_equal_to_pr25": control_matches,
    "posthoc_h6_personal_table_best_action_is_truly_favored": personal_target_correct,
    "posthoc_h6_personal_table_best_action_n": 32,
    "posthoc_table_selected_action_counts": target_counts,
    "diagnostic_not_a_model_run": True,
}
(art / "analysis/independent-check.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf8"
)
print(json.dumps(result, indent=2))
