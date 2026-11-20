"""Full fixed 4B/9B comparison of the minimal, self-generated prediction chain."""

import argparse
import time
from pathlib import Path

from experiments.frozen_capability.gpu import write
from experiments.frozen_capability.wire import strict_json

from .minimal import messages
from .staged_gpu import Runner as StagedRunner


class Runner(StagedRunner):
    make_messages = staticmethod(messages)

    def run(self):
        self.download()
        self.inference_deadline = min(
            self.deadline, time.time() + self.config["inference_seconds"]
        )
        lookup = {r["id"]: r for r in self.rows}
        outcome = "infrastructure_failure"
        for short in self.config["models"]:
            self.short = short
            model, tok, meta = self.load(short)
            versions = [p._version for p in model.parameters()]
            model_outcome = "infrastructure_failure"
            try:
                for job in self.config["jobs"]:
                    if (
                        self.inference_deadline - time.time()
                        < self.config["case_seconds"] + 15
                    ):
                        model_outcome = "budget_stop"
                        break
                    self.active = {
                        "model": short,
                        "case": job["id"],
                        "number": job["number"],
                        "arm": job["arm"],
                    }
                    self.status("generating", **self.active)
                    row = lookup[job["id"]]
                    result = (
                        self.generate(model, tok, meta, row["messages"])
                        if job["split"] == "startup"
                        else self.staged_case(model, tok, meta, row, job)
                    )
                    record = {**job, **result, "model": short, "thinking": False}
                    write(
                        self.out / short / "records" / f"{job['number']:03d}.json",
                        record,
                    )
                    self.results.append(
                        {
                            k: record[k]
                            for k in (
                                "model",
                                "number",
                                "id",
                                "split",
                                "arm",
                                "finish_reason",
                            )
                        }
                    )
                    write(self.out / "record-index.json", self.results)
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
                            model_outcome = "startup_failed"
                            break
                else:
                    model_outcome = "complete"
            finally:
                write(
                    self.out / short / "unchanged.json",
                    {
                        "parameter_versions_unchanged": versions
                        == [p._version for p in model.parameters()],
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "gradients_absent": all(
                            p.grad is None for p in model.parameters()
                        ),
                        "outcome": model_outcome,
                    },
                )
                del model, tok
                self.clear()
            if model_outcome != "complete":
                outcome = model_outcome
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
