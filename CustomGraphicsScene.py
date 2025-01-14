from PyQt6.QtWidgets import QGraphicsScene


class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent  # Referenz zur ImageViewerApp

    def mousePressEvent(self, event):
        """Verarbeitet Mausereignisse in der Szene."""
        if self.parent_widget:
            self.parent_widget.on_scene_mouse_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Verarbeitet Mausbewegungen in der Szene."""
        if self.parent_widget:
            self.parent_widget.on_scene_mouse_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Verarbeitet das Loslassen der Maus in der Szene."""
        if self.parent_widget:
            self.parent_widget.on_scene_mouse_release(event)
        super().mouseReleaseEvent(event)
