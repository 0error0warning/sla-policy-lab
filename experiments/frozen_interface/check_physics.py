"""Check the four-state answer key using a closed form, without the simulator."""

import argparse
import json
from decimal import Decimal, localcontext
from pathlib import Path


def verify(keys):
    rows, error = [], 0.0
    with localcontext() as ctx:
        ctx.prec = 50
        for name, key in keys.items():
            if key["arm"] != "body":
                continue
            computed = {}
            for word, cost, rate in zip(
                ("wait", "ask", "help", "trade"),
                ("0", ".25", "1.75", "1.75"),
                key["rates"],
            ):
                h, c = Decimal(str(key["health"])), Decimal(cost)
                action = max(Decimal(0), 700 + (h - 700) * (1 + c / 350))
                end = max(Decimal(0), 700 + (action - 700) * (Decimal(351) / 350) ** 4)
                maintained = None if word == "wait" else action
                p = Decimal(str(rate))
                death = (
                    Decimal(int(end == 0))
                    if maintained is None
                    else (1 - p) * int(end == 0) + p * int(maintained == 0)
                )
                values = {
                    "after_action": float(action),
                    "after_no_maintenance": float(end),
                    "after_maintenance": None
                    if maintained is None
                    else float(maintained),
                    "p_death": float(death),
                }
                for field, value in values.items():
                    if value is None:
                        assert key["actions"][word][field] is None
                    else:
                        error = max(error, abs(value - key["actions"][word][field]))
                computed[word] = values
            rows.append({"id": name, "closed_form_actions": computed})
    assert rows and error < 1e-12
    return {
        "method": "50-digit Decimal closed form: max(0,700+(h-700)*(1+c/350)); four unmaintained environmental events use power (351/350)^4. No simulator function called.",
        "states": len(rows),
        "max_absolute_error": error,
        "rows": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = verify(json.loads(a.key.read_bytes()))
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: result[k] for k in ("states", "max_absolute_error")}))
