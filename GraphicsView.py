from PyQt6.QtWidgets import QGraphicsView


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Standard auf Panning setzen

    def wheelEvent(self, event):
        """Zoom in/out mit dem Mausrad."""
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9
        old_zoom_factor = self.zoom_factor

        # Bestimme Zoomrichtung
        if event.angleDelta().y() > 0:
            self.zoom_factor *= zoom_in_factor
        else:
            self.zoom_factor *= zoom_out_factor

        # Skalieren
        scale_factor = self.zoom_factor / old_zoom_factor
        self.scale(scale_factor, scale_factor)
