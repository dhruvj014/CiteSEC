"""Generate BENCHMARKS.md from results/*.json and refresh the results block in README.md."""
from __future__ import annotations

import re

from citesec.config import load_config
from citesec.eval.report import README_END, README_START, readme_block, render


def update_readme(text: str, block: str) -> str:
    """Replace what sits between the RESULTS markers. Fails loudly if the markers are missing."""
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"README.md needs the markers {README_START} and {README_END}")
    return pattern.sub(lambda _: f"{README_START}\n{block}\n{README_END}", text)


def main() -> None:
    cfg = load_config()
    text = render(cfg.results_dir)
    (cfg.root / "BENCHMARKS.md").write_text(text, encoding="utf-8")
    print(f"wrote BENCHMARKS.md ({len(text.splitlines())} lines)")
    readme = cfg.root / "README.md"
    if readme.exists() and README_START in readme.read_text(encoding="utf-8"):
        readme.write_text(update_readme(readme.read_text(encoding="utf-8"),
                                        readme_block(cfg.results_dir)), encoding="utf-8")
        print("updated README.md results block")


if __name__ == "__main__":
    main()
