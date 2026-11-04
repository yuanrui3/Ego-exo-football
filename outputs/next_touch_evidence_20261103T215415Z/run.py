from __future__ import annotations

import argparse
import time

from common import RUN, read, write, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    import importlib
    names = {"0": ("common","preflight"), "1": ("evidence","clocks"), "2": ("evidence","reviews"), "3": ("evidence","contacts"), "4": ("experiments","targets_experiment"), "5": ("experiments","action_experiment"), "6": ("experiments","context_experiment"), "7": ("experiments","retrieval"), "8": ("external","external_study"), "9": ("followups","followups")}
    start = time.perf_counter()
    status(args.stage, "RUNNING", command=["run.py", "--stage", args.stage])
    try:
        module, name = names[args.stage]
        result = getattr(importlib.import_module(module), name)()
        write(RUN / f"stage_{args.stage}.json", result)
        status(args.stage, result.get("status", "COMPLETE"), seconds=time.perf_counter()-start)
    except Exception as exc:
        status(args.stage, "FAILED", seconds=time.perf_counter()-start, exception=type(exc).__name__, reason=str(exc))
        raise


if __name__ == "__main__":
    main()
