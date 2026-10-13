"""Answer evaluation driver. Stages are separate so each is resumable and re-runnable.

    python -m citesec.eval.run_answer_eval --split dev --stage retrieve   # GPU, no LLM
    python -m citesec.eval.run_answer_eval --split dev --stage generate   # LLM
    python -m citesec.eval.calibrate_gate                                  # dev only
    python -m citesec.eval.run_answer_eval --split dev --stage assemble
    python -m citesec.eval.run_answer_eval --split dev --stage judge      # LLM judge
    python -m citesec.eval.run_answer_eval --split dev --stage metrics

Test is for final numbers only. Re-running generation on test is refused once test answers exist
unless --overwrite-test is passed.
"""
from __future__ import annotations

import argparse
import logging

from citesec.config import load_config

STAGES = ["retrieve", "generate", "assemble", "judge", "metrics"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N items, for smoke tests")
    ap.add_argument("--overwrite-test", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    if args.split == "test" and args.stage in ("generate", "judge") and not args.overwrite_test:
        from citesec.eval.answer_data import answers_path

        if answers_path(cfg, "S2", "test").exists() and args.stage == "generate":
            raise SystemExit("test answers exist, pass --overwrite-test to regenerate")

    if args.stage == "retrieve":
        from citesec.eval import stage_retrieve

        stage_retrieve.run(cfg, args.split)
    elif args.stage == "generate":
        from citesec.eval import stage_generate

        stage_generate.run(cfg, args.split, args.limit)
    elif args.stage == "assemble":
        from citesec.eval import stage_assemble

        print(stage_assemble.run(cfg, args.split))
    elif args.stage == "judge":
        from citesec.eval import stage_judge

        stage_judge.run(cfg, args.split, args.limit)
    else:
        from citesec.eval import answer_metrics

        answer_metrics.run(cfg, args.split)


if __name__ == "__main__":
    main()
