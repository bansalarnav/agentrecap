#!/usr/bin/env python3

import json
import os
from pathlib import Path

import matplotlib.pyplot  # noqa: F401


cache_files = list(Path(os.environ["MPLCONFIGDIR"]).glob("fontlist*.json"))
if len(cache_files) != 1:
    raise SystemExit(f"Expected one Matplotlib font cache, found {len(cache_files)}")

cache_path = cache_files[0]
cache = json.loads(cache_path.read_text(encoding="utf-8"))
for font_list_name in ("afmlist", "ttflist"):
    cache[font_list_name] = [
        font
        for font in cache[font_list_name]
        if not Path(font["fname"]).is_absolute()
    ]
cache_path.write_text(json.dumps(cache), encoding="utf-8", newline="\n")
print(cache_path)
