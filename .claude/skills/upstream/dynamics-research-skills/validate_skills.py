#!/usr/bin/env python3
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def parse_frontmatter(text: str):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter delimited by ---")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise ValueError("Missing closing --- for YAML frontmatter")
    fm_lines = lines[1:end]
    data = {}
    current_key = None
    for line in fm_lines:
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = value.strip('"').strip("'")
                current_key = None
            else:
                current_key = key
                data[current_key] = {}
        elif current_key and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current_key][key.strip()] = value.strip().strip('"').strip("'")
        else:
            pass
    return data, end + 1


def validate_skill(skill_dir: Path):
    errors = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [f"{skill_dir}: missing SKILL.md"]
    text = skill_md.read_text(encoding="utf-8")
    try:
        data, body_start = parse_frontmatter(text)
    except Exception as e:
        return [f"{skill_dir.name}: invalid frontmatter: {e}"]

    name = data.get("name", "")
    desc = data.get("description", "")

    if not name:
        errors.append(f"{skill_dir.name}: missing required frontmatter field 'name'")
    if not desc:
        errors.append(f"{skill_dir.name}: missing required frontmatter field 'description'")

    if name:
        if len(name) > 64:
            errors.append(f"{skill_dir.name}: name longer than 64 chars")
        if not NAME_RE.fullmatch(name):
            errors.append(f"{skill_dir.name}: name must match ^[a-z0-9]+(?:-[a-z0-9]+)*$")
        if name != skill_dir.name:
            errors.append(f"{skill_dir.name}: frontmatter name '{name}' does not match directory name")

    if desc and len(desc) > 1024:
        errors.append(f"{skill_dir.name}: description longer than 1024 chars")

    line_count = len(text.splitlines())
    if line_count > 500:
        errors.append(f"{skill_dir.name}: SKILL.md has {line_count} lines; keep under ~500 recommended limit")

    body = text.splitlines()[body_start:]
    if not any(line.strip() for line in body):
        errors.append(f"{skill_dir.name}: SKILL.md body is empty")

    refs = skill_dir / "references"
    if refs.exists() and not refs.is_dir():
        errors.append(f"{skill_dir.name}: references exists but is not a directory")

    return errors


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    skill_dirs = sorted([p for p in root.iterdir() if p.is_dir() and (p / "SKILL.md").exists()])
    if not skill_dirs:
        print(f"No skill directories with SKILL.md found under {root}")
        sys.exit(1)

    all_errors = []
    for skill_dir in skill_dirs:
        all_errors.extend(validate_skill(skill_dir))

    if all_errors:
        print("Validation failed:")
        for err in all_errors:
            print(f" - {err}")
        sys.exit(1)

    print(f"Validated {len(skill_dirs)} skills successfully.")
    for skill_dir in skill_dirs:
        print(f" - {skill_dir.name}")


if __name__ == "__main__":
    main()
