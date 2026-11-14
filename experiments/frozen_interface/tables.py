"""Render compact result tables from verified CPU replay, without manual totals."""

import argparse
import gzip
import json
from pathlib import Path


def render(source):
    groups = json.loads((source / "summary.json").read_bytes())
    with gzip.open(source / "parsed-cases.jsonl.gz", "rt", encoding="utf8") as f:
        rows = [json.loads(line) for line in f]
    lines = [
        "| 模型 | 风险表输出形式 | 已执行 | 最终动作正确 | 两集合均正确 | 联合通过 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for g in groups:
        if g["arm"] != "table":
            continue
        c, n = g["counts"], g["planned"]
        sets = (
            f"{c['sets_consistent_with_own_risks']}/{n}"
            if g["form"] == "sets"
            else "不要求"
        )
        lines.append(
            f"| {g['model']} | {g['form']} | {c['executed']}/{n} | "
            f"{c['choice_correct']}/{n} | {sets} | {c['joint_success']}/{n} |"
        )
    lines += [
        "",
        "| 模型 | 模式 | 最终 JSON / 计划 | 全部数值正确 / 最终 JSON | 四动作风险均正确 / 最终 JSON | 动作正确 / 最终 JSON | 健康题任务 | 联合通过 / 计划 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in groups:
        if g["arm"] != "body":
            continue
        c, n = g["counts"], g["planned"]
        answered = c["final_json_parsed"]

        def ratio(key, c=c, answered=answered):
            return f"{c[key]}/{answered}" if answered else "无可评分最终 JSON"

        healthy = next(
            r
            for r in rows
            if r["model"] == g["model"]
            and r["thinking"] == g["thinking"]
            and r["arm"] == "body"
            and r["health"] == 90
        )
        health_result = (
            ("完成" if healthy["completed_task"] else "未完成任务")
            if healthy["final_json_parsed"]
            else "无可评分最终 JSON"
        )
        lines.append(
            f"| {g['model']} | {'thinking' if g['thinking'] else 'direct'} | "
            f"{answered}/{n} | {ratio('calculation_correct')} | "
            f"{ratio('risk_predictions_correct')} | {ratio('choice_correct')} | "
            f"{health_result} | {c['joint_success']}/{n} |"
        )
    lines += [
        "",
        "正确率的条件分母不能替代完整计划的联合门槛。没有最终答案、格式无效、算错与选错分别保留在 cases.csv。",
        "",
        "| 模型 | 模式 | 健康 | 结束原因 | 输出 token | 生成秒数 | 最终动作 | 环境中的死亡概率 | 最低可达风险 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["arm"] != "body":
            continue
        risk = "—" if r["actual_risk"] is None else f"{r['actual_risk']:.2f}"
        secs = "—" if r["elapsed_seconds"] is None else f"{r['elapsed_seconds']:.2f}"
        lines.append(
            f"| {r['model']} | {'thinking' if r['thinking'] else 'direct'} | "
            f"{r['health']:g} | {r['finish_reason'] or 'missing_record'} | "
            f"{r['generated_tokens'] if r['executed'] else '—'} | {secs} | "
            f"{r['choice'] or '—'} | {risk} | {min(r['reference']['risks'].values()):.2f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(render(a.source), encoding="utf8")
