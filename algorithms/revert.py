"""
版本还原脚本。

用法：
    python algorithms/revert.py            # 还原所有 *.v_original 文件
    python algorithms/revert.py <path>     # 还原指定文件

机制：
    扫描项目里所有 *.v_original 备份文件，逐个覆盖回原位并删除备份。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKUP_SUFFIX = ".v_original"


def revert_file(backup_path: Path) -> None:
    original_path = Path(str(backup_path).removesuffix(BACKUP_SUFFIX))
    if not backup_path.exists():
        print(f"[SKIP] 备份不存在: {backup_path}")
        return
    shutil.copy2(backup_path, original_path)
    backup_path.unlink()
    print(f"[REVERT] {original_path.relative_to(ROOT)}")


def revert_all() -> None:
    backups = list(ROOT.rglob(f"*{BACKUP_SUFFIX}"))
    # 排除 algorithms/ 目录下的备份（本来就不该有，但防御）
    backups = [b for b in backups if "algorithms" not in b.parts]

    if not backups:
        print("没有需要还原的备份。")
        return

    print(f"找到 {len(backups)} 个备份文件。")
    for b in backups:
        revert_file(b)
    print("全部还原完成。")


def main():
    if len(sys.argv) == 1:
        revert_all()
    elif len(sys.argv) == 2:
        path = Path(sys.argv[1])
        if not str(path).endswith(BACKUP_SUFFIX):
            path = Path(str(path) + BACKUP_SUFFIX)
        revert_file(path)
    else:
        print("用法: python algorithms/revert.py [path]")
        sys.exit(1)


if __name__ == "__main__":
    main()
