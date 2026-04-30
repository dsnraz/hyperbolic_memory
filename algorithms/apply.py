"""
统一的版本应用脚本。

用法：
    python algorithms/apply.py <version_name>

示例：
    python algorithms/apply.py v1_softplus_fix

机制：
    每个版本目录下都有一个 manifest.json，说明哪些文件替换哪些目标。
    本脚本读取后：
    1. 备份目标文件到 <target>.v_original（如果备份不存在）
    2. 用版本目录里的文件替换目标
    3. 打印替换清单

还原请用 revert.py。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # project root
BACKUP_SUFFIX = ".v_original"


def apply_version(version_name: str) -> None:
    version_dir = Path(__file__).resolve().parent / version_name
    if not version_dir.exists():
        print(f"[ERROR] 版本目录不存在: {version_dir}")
        sys.exit(1)

    manifest_path = version_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] 缺少 manifest: {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_map = manifest.get("files", {})  # {source_in_version_dir : target_relative_to_root}

    if not files_map:
        print(f"[WARN] manifest 里没有 files 字段，无需应用。")
        return

    print(f"=== 应用版本: {version_name} ===")
    print(f"版本说明: {manifest.get('description', '(无)')}")
    print()

    for src_rel, tgt_rel in files_map.items():
        src = version_dir / src_rel
        tgt = ROOT / tgt_rel

        if not src.exists():
            print(f"[SKIP] 版本文件不存在: {src}")
            continue
        if not tgt.exists():
            print(f"[WARN] 目标文件不存在: {tgt}（仍会创建）")

        backup = Path(str(tgt) + BACKUP_SUFFIX)
        if tgt.exists() and not backup.exists():
            shutil.copy2(tgt, backup)
            print(f"[BACKUP] {tgt.relative_to(ROOT)} -> {backup.name}")
        elif backup.exists():
            print(f"[SKIP-BACKUP] 备份已存在: {backup.name}（保留原 backup）")

        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
        print(f"[APPLY ] {src_rel} -> {tgt.relative_to(ROOT)}")

    print()
    print(f"=== 应用完成: {version_name} ===")
    instructions = manifest.get("post_apply_instructions", "")
    if instructions:
        print(instructions)


def main():
    if len(sys.argv) != 2:
        print("用法: python algorithms/apply.py <version_name>")
        sys.exit(1)
    apply_version(sys.argv[1])


if __name__ == "__main__":
    main()
