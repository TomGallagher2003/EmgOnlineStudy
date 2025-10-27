# docs/_scripts/gen_api.py
from __future__ import annotations

from pathlib import Path
import re
import mkdocs_gen_files

# Source roots to scan (kept as-is from your working version)
ROOTS = [
    Path("main.py"),
    Path("pages"),
    Path("workers"),
    Path("widgets"),
    Path("emg_default_settings.py"),
    Path("plotting"),
    Path("pipeline_sections")
]

# Regex: "#: some text" immediately above "NAME = ..."
DOC_ABOVE_CONST = re.compile(r"^#:\s*(.+)$")
CONST_ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=\s*")

def iter_modules():
    for root in ROOTS:
        if root.is_file() and root.suffix == ".py":
            yield root
        elif root.is_dir():
            for p in root.rglob("*.py"):
                if p.name == "__init__.py":
                    continue
                yield p

def find_documented_constants(src_path: Path) -> list[str]:
    """Return names of documented top-level constants (ALL_CAPS with a '#:' line just above)."""
    return []

# We will write pages under 'api/...'
# And write SUMMARY links RELATIVE to 'api/' (no 'api/' prefix in the link),
# then prefix with 'api/' when emitting SUMMARY.nav (so literate-nav reads absolute-from-docs-root paths).
summary_lines = ["# API Reference\n\n"]

for src in sorted(iter_modules()):
    module = src.with_suffix("").as_posix().replace("/", ".")                # e.g. core.ui.main_screen
    full_doc = Path("api", *src.with_suffix("").parts).with_suffix(".md")    # e.g. api/core/ui/main_screen.md
    rel_doc  = full_doc.relative_to("api").as_posix()  # e.g. core/ui/main_screen.md
    if rel_doc.startswith("pipeline_sections/models") and not rel_doc.endswith("cuda_check.md"):
        continue

    constants = find_documented_constants(src)

    # Write the page (virtual; parents auto-created)
    with mkdocs_gen_files.open(full_doc.as_posix(), "w") as fd:
        print(f"# {module}\n", file=fd)

        # Full module docs
        print(
            f"::: {module}\n"
            "    options:\n"
            "      show_source: true\n"
            "      show_if_no_docstring: true\n"
            "      members_order: source\n",
            file=fd,
        )

        # Auto-injected "Constants" subsection if we found documented ALL_CAPS
        if constants:
            print("\n## Constants\n", file=fd)
            const_members_block = "\n        - ".join(constants)
            print(
                f"::: {module}\n"
                "    options:\n"
                "      show_source: false\n"
                "      show_if_no_docstring: true\n"
                "      members_order: source\n"
                "      members:\n"
                f"        - {const_members_block}\n",
                file=fd,
            )

    # Edit link points to the real source
    mkdocs_gen_files.set_edit_path(full_doc.as_posix(), src.as_posix())

    # Add to SUMMARY with a link RELATIVE TO 'api/'
    summary_lines.append(f"- [{module}]({rel_doc})\n")

# Write SUMMARY inside 'api/' — links are relative to this file's folder,
# but we prefix with 'api/' so literate-nav treats them as absolute-from-docs-root.
summary_lines = [summary_lines[0]] + ["api/" + l for i, l in enumerate(summary_lines) if i > 0]
with mkdocs_gen_files.open("api/SUMMARY.nav", "w") as f:
    f.writelines(summary_lines)
