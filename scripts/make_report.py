"""Generate BENCHMARKS.md from results/*.json."""
from __future__ import annotations

from citesec.config import load_config
from citesec.eval.report import render


def main() -> None:
    cfg = load_config()
    text = render(cfg.results_dir)
    (cfg.root / "BENCHMARKS.md").write_text(text, encoding="utf-8")
    print(f"wrote BENCHMARKS.md ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
