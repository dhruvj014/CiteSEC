"""Create data/eval/split.json (dev 50, test 100). Never overwrites an existing split."""
from __future__ import annotations

from citesec.config import load_config
from citesec.eval.splits import ensure_split, split_path
from citesec.ingest.financebench import load_questions


def main() -> None:
    cfg = load_config()
    existed = split_path(cfg).exists()
    split = ensure_split(cfg, load_questions(cfg.fb_dir))
    print(("kept existing" if existed else "created"), split_path(cfg))
    print("dev", len(split["dev"]), split["dev_by_type"], "test", len(split["test"]))


if __name__ == "__main__":
    main()
