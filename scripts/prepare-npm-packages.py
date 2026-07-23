#!/usr/bin/env python3

import json
import re
import shutil
import stat
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "npm"
PLATFORMS = {
    "agentrecap-darwin-arm64": {"os": ["darwin"], "cpu": ["arm64"]},
    "agentrecap-darwin-x64": {"os": ["darwin"], "cpu": ["x64"]},
    "agentrecap-linux-arm64": {
        "os": ["linux"],
        "cpu": ["arm64"],
        "libc": ["glibc"],
    },
    "agentrecap-linux-x64": {
        "os": ["linux"],
        "cpu": ["x64"],
        "libc": ["glibc"],
    },
    "agentrecap-linux-arm64-musl": {
        "os": ["linux"],
        "cpu": ["arm64"],
        "libc": ["musl"],
    },
    "agentrecap-linux-x64-musl": {
        "os": ["linux"],
        "cpu": ["x64"],
        "libc": ["musl"],
    },
    "agentrecap-windows-arm64": {"os": ["win32"], "cpu": ["arm64"]},
    "agentrecap-windows-x64": {"os": ["win32"], "cpu": ["x64"]},
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
version = project["version"]
if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version):
    raise SystemExit(f"{version!r} is not a valid npm version")

for package_name, platform in PLATFORMS.items():
    package_dir = DIST / package_name
    binary_name = "agentrecap.exe" if "windows" in package_name else "agentrecap"
    binary = package_dir / "bin" / binary_name
    if not binary.is_file():
        raise SystemExit(f"Missing binary: {binary}")

    if binary_name == "agentrecap":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    write_json(
        package_dir / "package.json",
        {
            "name": package_name,
            "version": version,
            "description": f"Native {project['name']} binary for {package_name.removeprefix('agentrecap-')}",
            "repository": {
                "type": "git",
                "url": "https://github.com/bansalarnav/agentrecap",
            },
            "os": platform["os"],
            "cpu": platform["cpu"],
            **({"libc": platform["libc"]} if "libc" in platform else {}),
            "files": ["bin"],
            "preferUnplugged": True,
        },
    )

main_package_dir = DIST / "agentrecap"
main_package_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(ROOT / "npm" / "agentrecap" / "bin", main_package_dir / "bin", dirs_exist_ok=True)
shutil.copy2(ROOT / "npm" / "agentrecap" / "README.md", main_package_dir / "README.md")

main_manifest = json.loads(
    (ROOT / "npm" / "agentrecap" / "package.json").read_text(encoding="utf-8")
)
main_manifest["version"] = version
main_manifest["optionalDependencies"] = {
    package_name: version for package_name in PLATFORMS
}
write_json(main_package_dir / "package.json", main_manifest)

print(f"Prepared npm packages for agentrecap {version}")
