from __future__ import annotations

"""程序入口：启动 Qt 可视化工作台。"""

import os
from pathlib import Path


def _configure_runtime_environment() -> None:
    root = Path(__file__).resolve().parent

    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    qt_platforms = root / ".venv" / "Lib" / "site-packages" / "PyQt5" / "Qt5" / "plugins" / "platforms"
    if qt_platforms.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platforms))

    matplotlib_config = root / ".matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))


_configure_runtime_environment()

from ui_app import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    raise SystemExit(main())
#TODO:减面没实现  现在只能分成两个区域  没有实现自然过渡-混合场  指定方向性没实现   最后的精细裁剪是否太过严苛  
# 曲面边界  隐函数表示  
