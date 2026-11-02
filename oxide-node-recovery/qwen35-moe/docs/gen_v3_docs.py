"""Publish existing v3 Markdown without maintaining a second copy."""
from pathlib import Path
import os
import re
from urllib.parse import urlsplit

import mkdocs_gen_files

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "libs/infinity-v3"
DOCS_REF = os.environ.get("INFINITY_DOCS_REF", "main")
files = [SOURCE / name for name in (
    "README.md", "CONTRIBUTING.md", "LICENSING.md", "LICENSE", "NOTICE",
    "THIRD_PARTY_NOTICES.md", "CODE_OF_CONDUCT.md",
)]
files += [p for folder in ("docs", "LICENSES", "assets")
          for p in (SOURCE / folder).rglob("*") if p.is_file()]
published = set(files)

for source in files:
    destination = Path("v3") / source.relative_to(SOURCE)
    if source.suffix != ".md":
        with mkdocs_gen_files.open(destination, "wb") as output:
            output.write(source.read_bytes())
        continue

    def source_link(match):
        target = match.group(1)
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path or target.startswith("/"):
            return match.group(0)
        resolved = (source.parent / url.path).resolve()
        if resolved in published:
            return match.group(0)
        # Source code and vendor documentation remain browsable on GitHub.
        path = resolved.relative_to(REPO).as_posix()
        return f"](https://github.com/michaelfeil/infinity/blob/{DOCS_REF}/{path}" + (
            f"#{url.fragment}" if url.fragment else ""
        ) + ")"

    content = re.sub(r"\]\(([^\s)]+)\)", source_link, source.read_text())
    content = content.replace("<details>", '<details markdown="1">')
    if destination == Path("v3/README.md"):
        content = content.replace("# Infinity V3 ♾️", "# Infinity V3 ♾️\n\n"
            "> Documentation for the native v3 server. "
            "[Quickstart](quickstart.md) · "
            "[Python / v2 documentation](../cli_v2.md)", 1)
    with mkdocs_gen_files.open(destination, "w") as output:
        output.write(content)
    mkdocs_gen_files.set_edit_path(destination, f"../../{source.relative_to(REPO)}")
