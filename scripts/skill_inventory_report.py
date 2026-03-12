#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "when",
    "use",
    "into",
    "from",
    "that",
    "this",
    "should",
    "needs",
    "need",
    "current",
    "local",
    "project",
    "skill",
    "skills",
    "codex",
}


@dataclass
class SkillRecord:
    tree: str
    dir_name: str
    path: str
    frontmatter_name: str
    description: str
    has_openai_yaml: bool
    skill_hash: str
    openai_hash: str | None
    has_template_markers: bool


def parse_frontmatter(text: str) -> tuple[dict[str, str], int]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter delimited by ---")
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        raise ValueError("Missing closing --- for YAML frontmatter")
    frontmatter = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("Frontmatter must be a YAML mapping")
    return frontmatter, end + 1


def sha256_for_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_tokens(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in TOKEN_RE.findall(part.lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def load_skill_records(root: Path, tree: str) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    if not root.exists():
        return records
    for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        skill_text = skill_md.read_text(encoding="utf-8")
        data, _ = parse_frontmatter(skill_text)
        openai_yaml = skill_dir / "agents" / "openai.yaml"
        records.append(
            SkillRecord(
                tree=tree,
                dir_name=skill_dir.name,
                path=str(skill_dir),
                frontmatter_name=str(data.get("name", "")),
                description=str(data.get("description", "")),
                has_openai_yaml=openai_yaml.exists(),
                skill_hash=sha256_for_path(skill_md),
                openai_hash=sha256_for_path(openai_yaml) if openai_yaml.exists() else None,
                has_template_markers="TODO" in skill_text,
            )
        )
    return records


def build_report(agents_root: Path, claude_root: Path) -> dict[str, object]:
    agents_records = load_skill_records(agents_root, ".agents")
    claude_records = load_skill_records(claude_root, ".claude")

    agents_by_name = {record.dir_name: record for record in agents_records}
    claude_by_name = {record.dir_name: record for record in claude_records}

    all_names = sorted(set(agents_by_name) | set(claude_by_name))
    missing_in_agents = [name for name in all_names if name not in agents_by_name]
    missing_in_claude = [name for name in all_names if name not in claude_by_name]

    frontmatter_issues: list[dict[str, str]] = []
    for record in agents_records + claude_records:
        if record.frontmatter_name != record.dir_name:
            frontmatter_issues.append(
                {
                    "tree": record.tree,
                    "skill": record.dir_name,
                    "issue": f"frontmatter name '{record.frontmatter_name}' does not match directory name",
                }
            )
        elif not NAME_RE.fullmatch(record.frontmatter_name):
            frontmatter_issues.append(
                {
                    "tree": record.tree,
                    "skill": record.dir_name,
                    "issue": "frontmatter name does not match required naming rule",
                }
            )

    content_drift: list[dict[str, str]] = []
    metadata_drift: list[dict[str, str]] = []
    template_debt: list[dict[str, str]] = []
    for name in sorted(set(agents_by_name) & set(claude_by_name)):
        agents_record = agents_by_name[name]
        claude_record = claude_by_name[name]
        if agents_record.skill_hash != claude_record.skill_hash:
            content_drift.append({"skill": name, "agents_path": agents_record.path, "claude_path": claude_record.path})
        if agents_record.openai_hash != claude_record.openai_hash:
            metadata_drift.append({"skill": name, "agents_path": agents_record.path, "claude_path": claude_record.path})

    for record in agents_records + claude_records:
        if record.has_template_markers:
            template_debt.append({"tree": record.tree, "skill": record.dir_name, "path": record.path})

    merge_hints: list[dict[str, object]] = []
    for left, right in combinations(agents_records, 2):
        similarity = jaccard_similarity(
            normalize_tokens(left.dir_name, left.description),
            normalize_tokens(right.dir_name, right.description),
        )
        if similarity >= 0.55:
            merge_hints.append(
                {
                    "left": left.dir_name,
                    "right": right.dir_name,
                    "similarity": round(similarity, 3),
                }
            )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "agents_root": str(agents_root),
        "claude_root": str(claude_root),
        "counts": {
            "agents": len(agents_records),
            "claude": len(claude_records),
        },
        "missing_in_agents": missing_in_agents,
        "missing_in_claude": missing_in_claude,
        "frontmatter_issues": frontmatter_issues,
        "content_drift": content_drift,
        "metadata_drift": metadata_drift,
        "template_debt": template_debt,
        "merge_hints": merge_hints,
        "skills": {
            "agents": [asdict(record) for record in agents_records],
            "claude": [asdict(record) for record in claude_records],
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    counts = report["counts"]
    missing_in_agents = report["missing_in_agents"]
    missing_in_claude = report["missing_in_claude"]
    frontmatter_issues = report["frontmatter_issues"]
    content_drift = report["content_drift"]
    metadata_drift = report["metadata_drift"]
    template_debt = report["template_debt"]
    merge_hints = report["merge_hints"]

    lines = [
        "# Skill Inventory Report",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- .agents skills: {counts['agents']}",
        f"- .claude skills: {counts['claude']}",
        "",
        "## Mirror Gaps",
    ]
    if missing_in_agents or missing_in_claude:
        if missing_in_agents:
            lines.append(f"- missing in .agents: {', '.join(missing_in_agents)}")
        if missing_in_claude:
            lines.append(f"- missing in .claude: {', '.join(missing_in_claude)}")
    else:
        lines.append("- none")

    lines.extend(["", "## Frontmatter Issues"])
    if frontmatter_issues:
        for issue in frontmatter_issues:
            lines.append(f"- {issue['tree']} / {issue['skill']}: {issue['issue']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Drift"])
    if content_drift:
        for item in content_drift:
            lines.append(f"- content drift: {item['skill']}")
    else:
        lines.append("- content drift: none")
    if metadata_drift:
        for item in metadata_drift:
            lines.append(f"- metadata drift: {item['skill']}")
    else:
        lines.append("- metadata drift: none")

    lines.extend(["", "## Template Debt"])
    if template_debt:
        for item in template_debt:
            lines.append(f"- {item['tree']} / {item['skill']}: TODO markers still present")
    else:
        lines.append("- none")

    lines.extend(["", "## Merge Hints"])
    if merge_hints:
        for item in merge_hints:
            lines.append(f"- {item['left']} <-> {item['right']} (similarity={item['similarity']})")
    else:
        lines.append("- none above heuristic threshold")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local skill trees for mirror drift and merge hints.")
    parser.add_argument("--agents-root", default=".agents/skills/project", help="Path to the .agents project skills root.")
    parser.add_argument("--claude-root", default=".claude/skills/project", help="Path to the .claude project skills root.")
    parser.add_argument("--json-out", help="Optional path for machine-readable JSON output.")
    parser.add_argument("--md-out", help="Optional path for Markdown output.")
    args = parser.parse_args()

    report = build_report(Path(args.agents_root), Path(args.claude_root))
    markdown = render_markdown(report)

    print(markdown, end="")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.md_out:
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
