#!/usr/bin/env python3
"""Rewrite fixture pixel paths for a staged remote directory.

Usage: rewrite_fixtures.py INPUT.json FIXTURE_DIR OUTPUT.json
"""
import json
import sys
from pathlib import Path

source, target, output = map(Path, sys.argv[1:4])
data = json.loads(source.read_text())
for fixture in data["fixtures"]:
    for image in fixture.get("images", []):
        image["pixels"] = str(target / Path(image["pixels"]).name)
output.write_text(json.dumps(data, indent=2) + "\n")
print(output)
