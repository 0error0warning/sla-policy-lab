"""Within-runtime monolithic versus self-generated staged prediction."""

import argparse
import json
import time
from pathlib import Path

from experiments.frozen_capability.gpu import digest, write
from experiments.frozen_capability.wire import strict_json

from .gpu import Runner as BaseRunner
from .staged import STAGE_CAPS, assemble, messages


class Runner(BaseRunner):
    make_messages = staticmethod(messages)

    def wait_gate(self):
        hashes = {
            r["id"]: digest(self.out / "4B/records" / f"{r['number']:03d}.json")
            for r in self.results
            if r["split"] == "development" and r["arm"] == "staged"
        }
        write(self.out / "development-ready.json", hashes)
        self.status("await_development_gate")
        expires = min(self.inference_deadline - 60, time.time() + 300)
        while time.time() < expires:
            path = self.root / "development-gate.json"
            if path.exists():
                gate = json.loads(path.read_bytes())
                assert gate["records"] == hashes
                assert (
                    gate["public_prompts_sha256"]
                    == self.config["public_prompts_sha256"]
                )
                assert type(gate["passed"]) is bool
                write(self.out / "development-gate.json", gate)
                return gate["passed"]
            time.sleep(1)
        raise RuntimeError("external development gate unavailable")

    def staged_case(self, model, tok, meta, row, job):
        outputs, calls = {}, []
        started = time.monotonic()
        for stage, cap in STAGE_CAPS.items():
            remaining = self.config["case_seconds"] - (time.monotonic() - started)
            if remaining < 1:
                break
            # P and Q do not copy each other's forecast, only their own extracted effects.
            previous = (
                outputs
                if stage == "choice"
                else {k: v for k, v in outputs.items() if k == "effects"}
            )
            prompt = self.make_messages(row, stage, previous)
            self.status("generating", **self.active, substage=stage)
            result = self.generate(
                model, tok, meta, prompt, max_tokens=cap, seconds=remaining
            )
            raw = {
                **result,
                "id": job["id"],
                "arm": job["arm"],
                "stage": stage,
                "messages": prompt,
            }
            write(
                self.out
                / getattr(self, "short", "4B")
                / "stages"
                / f"{job['number']:03d}-{stage}.json",
                raw,
            )
            calls.append(raw)
            outputs[stage] = result["final_text"]
            if result["completion_status"] != "completed":
                break
        complete = len(calls) == 4 and all(
            r["completion_status"] == "completed" for r in calls
        )
        status = (
            "completed"
            if complete
            else calls[-1]["completion_status"]
            if calls
            else "time_limit"
        )
        if not complete and status == "completed":
            status = "time_limit"
        return {
            "completion_status": status,
            "finish_reason": "eos" if complete else status,
            "final_text": json.dumps(assemble(outputs)) if complete else None,
            "assembled_from_model_outputs": True,
            "stages": outputs,
            "calls": len(calls),
            "generated_tokens": sum(r["generated_tokens"] for r in calls),
            "elapsed_seconds": time.monotonic() - started,
        }

    def run(self):
        self.download()
        model, tok, meta = self.load("4B")
        versions = [p._version for p in model.parameters()]
        self.inference_deadline = min(
            self.deadline, time.time() + self.config["inference_seconds"]
        )
        outcome = "infrastructure_failure"
        try:
            lookup = {r["id"]: r for r in self.rows}
            gated = False
            for job in self.config["jobs"]:
                if job["split"] == "confirmation" and not gated:
                    gated = True
                    if not self.wait_gate():
                        outcome = "development_failed"
                        break
                if (
                    self.inference_deadline - time.time()
                    < self.config["case_seconds"] + 15
                ):
                    outcome = "budget_stop"
                    break
                self.active = {
                    "case": job["id"],
                    "number": job["number"],
                    "arm": job["arm"],
                }
                self.status("generating", **self.active)
                row = lookup[job["id"]]
                result = (
                    self.staged_case(model, tok, meta, row, job)
                    if job["arm"] == "staged"
                    else self.generate(model, tok, meta, row["messages"])
                )
                record = {**job, **result, "model": "4B", "thinking": False}
                write(self.out / "4B/records" / f"{job['number']:03d}.json", record)
                self.results.append(
                    {
                        k: record[k]
                        for k in ("number", "id", "split", "arm", "finish_reason")
                    }
                )
                self.status("case_saved", **self.active)
                assert versions == [p._version for p in model.parameters()]
                if job["split"] == "startup":
                    truth = (
                        {"ready": True}
                        if job["id"] == "startup-1"
                        else {"check": {"values": [7, 0, None], "flag": False}}
                    )
                    if (
                        result["completion_status"] != "completed"
                        or strict_json(result["final_text"]) != truth
                    ):
                        outcome = "startup_failed"
                        break
            else:
                outcome = "complete"
            write(
                self.out / "completed.json",
                {
                    "outcome": outcome,
                    "records": self.results,
                    "case_records": len(self.results),
                    "all_parameters_frozen": True,
                },
            )
            self.status(outcome)
        finally:
            write(
                self.out / "4B/unchanged.json",
                {
                    "parameter_versions_unchanged": versions
                    == [p._version for p in model.parameters()],
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "gradients_absent": all(p.grad is None for p in model.parameters()),
                    "outcome": outcome,
                },
            )
            del model, tok
            self.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runner = Runner(args.root)
    try:
        runner.run()
    except Exception as error:
        runner.status("stopped", error_type=type(error).__name__, reason=str(error))
        raise
