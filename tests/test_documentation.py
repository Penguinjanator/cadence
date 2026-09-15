"""Run the reader's actual introductory snippets and check local documentation links."""

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "page",
    ["docs/quickstart.md", "docs/memory.md", "docs/continuous.md"],
)
def test_introductory_python_snippets(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "documentation_example"}
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)```", (ROOT / page).read_text(), re.S)
    ):
        exec(compile(code, f"{page}:python-block-{index + 1}", "exec"), namespace)
    if page == "docs/quickstart.md":
        assert namespace["learner"].updates == 1
        assert (tmp_path / "predictive_circuit.npz").exists()


def test_local_documentation_links_resolve():
    problems = []
    for page in [ROOT / "README.md", ROOT / "examples/README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", page.read_text()):
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = unquote(target).partition("#")
            destination = (page.parent / path).resolve() if path else page
            if not destination.exists():
                problems.append(f"{page.name}: missing {target}")
            elif anchor and destination.suffix == ".md":
                content = destination.read_text()
                headings = re.findall(r"^#+\s+(.+)$", content, re.M)
                slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                if anchor not in slugs and f'id="{anchor}"' not in content:
                    problems.append(f"{page.name}: missing anchor {target}")
    assert not problems, "\n".join(problems)


def test_core_has_only_the_two_experience_examples():
    assert {p.name for p in (ROOT / "examples").glob("*.py")} == {
        "experience.py", "generic_brain.py",
    }
