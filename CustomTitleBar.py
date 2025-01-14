from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QColor, QIcon


class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # Layout der Titelleiste
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Titel-Label
        self.title_label = QLabel("Region Malen: Linie, Rechteck oder Kreis", self)
        self.title_label.setStyleSheet("color: white; font-size: 14px; padding-left: 10px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.layout.addWidget(self.title_label)

        # Buttons für Minimieren, Maximieren und Schließen
        self.minimize_button = QPushButton("_")
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.setStyleSheet("background-color: #1f1f1f; color: white; border: none;")
        self.minimize_button.clicked.connect(self.minimize_window)
        self.layout.addWidget(self.minimize_button)

        self.maximize_button = QPushButton("[ ]")
        self.maximize_button.setFixedSize(30, 30)
        self.maximize_button.setStyleSheet("background-color: #1f1f1f; color: white; border: none;")
        self.maximize_button.clicked.connect(self.maximize_window)
        self.layout.addWidget(self.maximize_button)

        self.close_button = QPushButton("X")
        self.close_button.setFixedSize(30, 30)
        self.close_button.setStyleSheet("background-color: #d32f2f; color: white; border: none;")
        self.close_button.clicked.connect(self.close_window)
        self.layout.addWidget(self.close_button)

        # Hintergrundfarbe der Titelleiste
        self.setStyleSheet("background-color: #121212;")

        # Initialisiere Drag-Position
        self.drag_position = None

    def mousePressEvent(self, event):
        """Ermöglicht das Ziehen des Fensters."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Aktualisiert die Fensterposition beim Ziehen."""
        if self.drag_position and event.buttons() == Qt.MouseButton.LeftButton:
            self.parent.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Beendet das Ziehen des Fensters."""
        self.drag_position = None
        event.accept()

    def minimize_window(self):
        """Minimiert das Fenster."""
        self.parent.showMinimized()

    def maximize_window(self):
        """Maximiert oder stellt das Fenster wieder her."""
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def close_window(self):
        """Schließt das Fenster."""
        self.parent.close()
