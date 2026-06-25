"""
启动对话框 - 选择创建新工程或打开已有工程
"""

from __future__ import annotations

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from domain.project.project_manager import ProjectManager


class StartupDialog(QtWidgets.QDialog):
    """启动对话框"""
    
    def __init__(self, project_manager: ProjectManager, parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.selected_action = None  # "new", "open", "recent"
        self.selected_project_path = None
        
        self.setWindowTitle("鞋底晶格生成系统")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        # 隐藏默认标题栏
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        
        # 用于窗口拖动
        self._drag_pos = None
        
        # 设置深色背景
        self.setStyleSheet("""
            QDialog {
                background-color: #2c2c2c;
            }
        """)
        
        self._build_ui()
    
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 自定义标题栏
        title_bar = QtWidgets.QWidget()
        title_bar.setStyleSheet("background-color: #1e1e1e;")
        title_bar.setFixedHeight(40)
        title_bar_layout = QtWidgets.QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(15, 0, 5, 0)
        title_bar_layout.setSpacing(0)
        
        # 标题
        self.title_label = QtWidgets.QLabel("鞋底晶格生成系统")
        self.title_label.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold;")
        title_bar_layout.addWidget(self.title_label)
        title_bar_layout.addStretch()
        
        # 关闭按钮
        btn_close = QtWidgets.QPushButton("✕")
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #ffffff;
                font-size: 16px;
                padding: 6px 16px;
                min-width: 46px;
                max-width: 46px;
                min-height: 32px;
                max-height: 32px;
            }
            QPushButton:hover {
                background-color: #e81123;
            }
            QPushButton:pressed {
                background-color: #c50f1f;
            }
        """)
        btn_close.clicked.connect(self.reject)
        title_bar_layout.addWidget(btn_close)
        
        layout.addWidget(title_bar)
        
        # 内容区域
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(30, 30, 30, 30)
        
        # 欢迎标题
        welcome_title = QtWidgets.QLabel("欢迎使用")
        welcome_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        welcome_title.setAlignment(QtCore.Qt.AlignCenter)
        content_layout.addWidget(welcome_title)
        
        # 副标题
        subtitle = QtWidgets.QLabel("请选择一个选项开始")
        subtitle.setStyleSheet("font-size: 13px; color: #aaaaaa;")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        content_layout.addWidget(subtitle)
        
        content_layout.addSpacing(10)
        
        # 按钮样式 - 深色主题
        button_style = """
            QPushButton {
                background-color: #3a3a3a;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                padding: 15px;
                text-align: left;
                font-size: 14px;
                color: #ffffff;
            }
            QPushButton:hover {
                border-color: #2196F3;
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """
        
        # 新建工程按钮
        btn_new = QtWidgets.QPushButton("📄  创建新工程")
        btn_new.setStyleSheet(button_style)
        btn_new.setMinimumHeight(60)
        btn_new.setCursor(QtCore.Qt.PointingHandCursor)
        btn_new.clicked.connect(self._on_new_project)
        content_layout.addWidget(btn_new)
        
        # 打开工程按钮
        btn_open = QtWidgets.QPushButton("📂  打开已有工程...")
        btn_open.setStyleSheet(button_style)
        btn_open.setMinimumHeight(60)
        btn_open.setCursor(QtCore.Qt.PointingHandCursor)
        btn_open.clicked.connect(self._on_open_project)
        content_layout.addWidget(btn_open)
        
        # 最近工程
        recent_projects = self.project_manager.get_recent_projects()
        if recent_projects:
            content_layout.addSpacing(10)
            
            recent_label = QtWidgets.QLabel("最近工程:")
            recent_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #aaaaaa;")
            content_layout.addWidget(recent_label)
            
            # 最近工程列表（最多显示5个）
            for project_path in recent_projects[:5]:
                btn_recent = QtWidgets.QPushButton(f"🕒  {project_path.name}")
                btn_recent.setStyleSheet(button_style)
                btn_recent.setMinimumHeight(45)
                btn_recent.setCursor(QtCore.Qt.PointingHandCursor)
                btn_recent.setToolTip(str(project_path))
                btn_recent.clicked.connect(lambda checked, p=project_path: self._on_open_recent(p))
                content_layout.addWidget(btn_recent)
        
        content_layout.addStretch()
        
        # 底部按钮
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addStretch()
        
        btn_cancel = QtWidgets.QPushButton("取消")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 8px 20px;
                color: #ffffff;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        btn_cancel.setMinimumWidth(80)
        btn_cancel.clicked.connect(self.reject)
        bottom_layout.addWidget(btn_cancel)
        
        content_layout.addLayout(bottom_layout)
        
        layout.addWidget(content_widget)
    
    def _on_new_project(self):
        """创建新工程"""
        self.selected_action = "new"
        self.accept()
    
    def _on_open_project(self):
        """打开工程"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "打开工程",
            str(Path.home()),
            "工程文件 (*.slp);;所有文件 (*.*)"
        )
        
        if file_path:
            self.selected_action = "open"
            self.selected_project_path = Path(file_path)
            self.accept()
    
    def _on_open_recent(self, project_path: Path):
        """打开最近工程"""
        if not project_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "文件不存在",
                f"工程文件不存在:\n{project_path}"
            )
            return
        
        self.selected_action = "recent"
        self.selected_project_path = project_path
        self.accept()
    
    def mousePressEvent(self, event):
        """鼠标按下事件 - 用于拖动窗口"""
        if event.button() == QtCore.Qt.LeftButton:
            # 检查是否点击在标题栏区域（前40像素高度）
            if event.pos().y() <= 40:
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 拖动窗口"""
        if event.buttons() == QtCore.Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self._drag_pos = None
        super().mouseReleaseEvent(event)
