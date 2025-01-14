import sys
from PyQt6.QtCore import QFile, QIODevice
import cv2
import json
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
    QFileDialog, QListWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSlider, QLabel, QComboBox,
    QListWidgetItem, QMessageBox, QInputDialog
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QIcon
from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, QFile
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.models import load_model


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


#from CustomTitleBar import CustomTitleBar
dark_stylesheet = """
QMainWindow {
    background-color: #121212;  /* Dunkler Hintergrund für das Hauptfenster */
    color: #ffffff;  /* Weißer Text */
}

QWidget {
    background-color: #121212;
    color: #ffffff;
    font-size: 14px;
    font-family: Arial, Helvetica, sans-serif;
}

QPushButton {
    background-color: #1f1f1f;
    border: 1px solid #3e3e3e;
    border-radius: 4px;
    color: #ffffff;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #3e3e3e;
    border: 1px solid #ffffff;
}

QPushButton:pressed {
    background-color: #555555;
}

QComboBox {
    background-color: #1f1f1f;
    border: 1px solid #3e3e3e;
    border-radius: 4px;
    padding: 6px;
    color: #ffffff;
}

QListWidget {
    background-color: #1f1f1f;
    border: 1px solid #3e3e3e;
    border-radius: 4px;
    padding: 4px;
    color: #ffffff;
}
QListWidget::item:hover {
    background-color: #3e3e3e;
}
QListWidget::item:selected {
    background-color: #555555;
}

QGraphicsView {
    border: 1px solid #3e3e3e;
    background-color: #181818;
}

QSlider::groove:horizontal {
    border: 1px solid #3e3e3e;
    height: 8px;
    background: #1f1f1f;
}
QSlider::handle:horizontal {
    background: #555555;
    border: 1px solid #3e3e3e;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #777777;
}

QLabel {
    color: #ffffff;
}

QToolTip {
    background-color: #222222;
    color: #ffffff;
    border: 1px solid #555555;
}
"""

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


class ImageViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stylesheet_filename = "qss/dark_theme.qss"
        #self.loadStylesheet(self.stylesheet_filename)
        #self.setWindowTitle("Region Malen: Linie, Rechteck oder Kreis")
        self.setGeometry(100, 100, 1200, 800)

        #titelbar
        # Entferne die native Titelleiste
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Haupt-Widget
        self.central_widget = QWidget(self)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Benutzerdefinierte Titelleiste hinzufügen
        self.title_bar = CustomTitleBar(self)
        self.layout.addWidget(self.title_bar)

        # Hauptinhalt (Dummy-Widget hier für die Demonstration)
       # self.main_content = QLabel("Hauptinhalt", self)
        #self.main_content.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.main_content.setStyleSheet("background-color: #1f1f1f; color: white; font-size: 18px;")
        #self.layout.addWidget(self.main_content)

        self.setCentralWidget(self.central_widget)

        # Pfade und Parameter
        self.json_file_path = "./Labels/labels.json" # Pfad zur JSON-Datei mit Labels
        self.image_size = (128, 128)  # Größe der Bilder für das Modell
        self.batch_size = 32  # Batch-Größe
        self.num_epochs = 10  # Anzahl der Epoche


        # Hauptvariablen
        self.selected_Image_path=None
        self.images_path=None
        self.image_paths = []
        self.original_image = None
        self.labels = {}  # Dictionary, um die Labels der Bilder zu speichern
        self.current_label = None  # Aktuell ausgewähltes Label
        self.scene = CustomGraphicsScene(self)
        self.pixmap_item = None
        self.drawing = False
        self.start_point = None
        self.last_point = None
        self.mask = None
        self.line_width = 3  # Standardlinienbreite
        self.drawing_mode = "Line"  # Standardmodus: "Line"
        self.temp_item = None  # Temporäres Element für Live-Anzeige
        self.undo_stack = []  # Stack für Rückgängig-Funktion
        self.region_points = []
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.coco_data = self.create_coco_structure()

        # UI-Komponenten
       # self.central_widget = QWidget()
        #self.setCentralWidget(self.central_widget)

        self.Main_layout = QHBoxLayout()

        # Linke Bildliste
        self.image_list = QListWidget()
        self.image_list.setFixedWidth(200)
        self.image_list.itemClicked.connect(self.load_selected_image)
        self.Main_layout.addWidget(self.image_list)

        # Rechte Hauptansicht
        self.right_layout = QVBoxLayout()

        # Bildanzeige (QGraphicsView)
        self.view = GraphicsView()
        self.view.setScene(self.scene)
        self.right_layout.addWidget(self.view)

        # Steuerbuttons
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Bilder laden")
        self.reset_button = QPushButton("Zurücksetzen")
        self.undo_button = QPushButton("Rückgängig")  # Neue Schaltfläche für Rückgängig
        self.save_button = QPushButton("Maske speichern")
        self.invert_button = QPushButton("Maske umkehren")  # Neue Schaltfläche zum Umkehren der Maske
        self.label_save_button = QPushButton("Label speichern")
        self.label_export_button = QPushButton("Labels exportieren")
        self.dataset_creat_button = QPushButton("Dataset erstellen")
        self.train_dataset = QPushButton("Modell trainieren")


        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.reset_button)
        self.button_layout.addWidget(self.undo_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.invert_button)
        self.button_layout.addWidget(self.label_save_button)
        self.button_layout.addWidget(self.label_export_button)
        self.button_layout.addWidget(self.dataset_creat_button)
        self.button_layout.addWidget(self.train_dataset)

        self.right_layout.addLayout(self.button_layout)

        # Label-Management
        self.label_layout = QHBoxLayout()
        self.label_selector = QComboBox()
        self.label_selector.setEditable(True)  # Benutzer kann neue Labels hinzufügen
        self.label_selector.addItems(["Label 1", "Label 2", "Label 3"])  # Beispiel-Labels
        self.label_selector.currentTextChanged.connect(self.update_current_label)
        self.label_add_button = QPushButton("Label hinzufügen")
        self.label_add_button.clicked.connect(self.add_new_label)
        self.label_layout.addWidget(QLabel("Label:"))
        self.label_layout.addWidget(self.label_selector)
        self.label_layout.addWidget(self.label_add_button)
        self.right_layout.addLayout(self.label_layout)



        # Dropdown für Modusauswahl
        self.mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Modus:")
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Line", "Rectangle", "Circle", "Eraser"])
        self.mode_selector.currentTextChanged.connect(self.update_drawing_mode)

        self.mode_layout.addWidget(self.mode_label)
        self.mode_layout.addWidget(self.mode_selector)
        self.right_layout.addLayout(self.mode_layout)

        # Slider für Linienbreite
        self.slider_layout = QHBoxLayout()
        self.slider_label = QLabel("Linienbreite:")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(self.line_width)
        self.slider.valueChanged.connect(self.update_line_width)

        self.slider_layout.addWidget(self.slider_label)
        self.slider_layout.addWidget(self.slider)
        self.right_layout.addLayout(self.slider_layout)

        self.Main_layout.addLayout(self.right_layout)

        self.layout.addLayout(self.Main_layout)

        # Signal-Verbindungen
        self.load_button.clicked.connect(self.load_images)
        self.reset_button.clicked.connect(self.reset_view)
        self.undo_button.clicked.connect(self.undo_last_action)  # Verbindung der Rückgängig-Funktion
        self.save_button.clicked.connect(self.save_mask)
        self.invert_button.clicked.connect(self.invert_and_show_mask)  # Verbindung der neuen Schaltfläche
        self.label_save_button.clicked.connect(self.save_label)
        self.label_export_button.clicked.connect(self.export_labels)
        self.dataset_creat_button.clicked.connect(self.creat_Label_and_dataset)
        self.train_dataset.clicked.connect(self.trainModel)

    def get_all_json_files(self, directory):
        """Gibt eine Liste aller JSON-Dateien in einem Verzeichnis zurück."""
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        return json_files
    # 1. Lade die Labels und Bildpfade aus der JSON-Datei
    def load_labels(self, json_file):
        """Lädt die Labels und Bildpfade aus der JSON-Datei."""
        with open(json_file, "r") as f:
            labels = json.load(f)
        return labels




        # 2. Bilder und Labels in Arrays konvertieren
    def prepare_data(self, labels_dict, image_size):
        """Bereitet die Bilder und Labels für das Training vor."""
        images = []
        labels = []
        label_map = {}  # Label zu Index-Mapping
        label_counter = 0

        for file_name, label in labels_dict.items():



            file_path = os.path.join(self.images_path, file_name)  # Passe den Ordner an

            # Lade das Bild
            image = cv2.imread(file_path)
            if image is None:
                print(f"Bild konnte nicht geladen werden: {file_path}")
                continue

            # BGR -> RGB und Resize
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, image_size)
            images.append(image)

            # Label in einen Index umwandeln
            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1
            labels.append(label_map[label])

        return np.array(images), np.array(labels), label_map

    def load_json(self, file_path):
        """Lädt eine JSON-Datei und gibt ein leeres Dictionary zurück, falls Fehler auftreten."""
        #jsonfile = self.get_all_json_files(file_path)
       #print(jsonfile)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Fehler beim Lesen der JSON-Datei {file_path}: {e}")
                return {}
        else:
            print(f"Datei {file_path} existiert nicht.")
            return {}


    def trainModel(self):
        # Modell laden
        model = load_model("image_classifier.h5")
        # Label-Mapping
        label_map = {0: "Katze", 1: "Schmitterling"}  # Passe dies an dein Mapping an
        # Vorhersage
        image_path = self.selected_Image_path
        image = self.preprocess_image(image_path, (128, 128))
        prediction = model.predict(image)
        predicted_label = label_map[np.argmax(prediction)]
        # Zeige das vorhergesagte Label als Overlay
        self.overlay_image = self.add_overlay(self.original_image, predicted_label)

         # Bildgröße überprüfen und skalieren
        max_dim = 1000  # Maximale Breite oder Höhe
        h, w = self.overlay_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            self.overlay_image = cv2.resize(self.overlay_image, (int(w * scale), int(h * scale)),
                                             interpolation=cv2.INTER_AREA)
        # Bild als QPixmap laden
        try:
            qimage = QImage(self.overlay_image.data, self.overlay_image.shape[1], self.overlay_image.shape[0],
                            QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"Fehler beim Konvertieren des Bildes: {e}")
            return
        # Bild zur Szene hinzufügen
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)
        self.undo_stack.clear()  # Stack zurücksetzen



    def add_overlay(self, image, label):
        """Fügt ein halbtransparentes Overlay mit dem vorhergesagten Label zum Bild hinzu."""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        # Overlay-Hintergrund (transparent)
        overlay_color = (255, 0, 0)  # Rot (RGB)
        alpha = 0.5  # Transparenz

        # Rechteck für das Label
        rect_start = (10, h - 50)
        rect_end = (w - 10, h - 10)
        cv2.rectangle(overlay, rect_start, rect_end, overlay_color, -1)

        # Text auf das Rechteck schreiben
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_color = (255, 255, 255)  # Weiß
        cv2.putText(overlay, label, (20, h - 20), font, font_scale, text_color, font_thickness)

        # Bild mit Overlay kombinieren
        combined_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return combined_image

    # Bild vorbereiten
    def preprocess_image(self,image_path, image_size):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = np.expand_dims(image, axis=0)  # Batch-Dimension hinzufügen
        return image


    #Creat Label And Dataset
    def creat_Label_and_dataset(self):
        # Lade die Labels
        labels_dict = self.load_labels(self.json_file_path)
        # Bereite die Daten vor
        self.images, self.labels, self.label_map = self.prepare_data(labels_dict, self.image_size)
        print(f"Bilder geladen: {len(self.images)}")
        print(f"Label-Mapping: {self.label_map}")

            # Teile die Daten in Training und Test auf
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)

        # TensorFlow-Datensätze erstellen
        train_dataset = self.create_tf_dataset(X_train, y_train, self.batch_size)
        test_dataset = self.create_tf_dataset(X_test, y_test, self.batch_size)

        # Modell erstellen
        num_classes = len(self.label_map)
        model = self.create_model(input_shape=(self.image_size[0], self.image_size[1], 3), num_classes=num_classes)

        # Modell trainieren
        model.fit(train_dataset, epochs=self.num_epochs, validation_data=test_dataset)

        # Modell speichern
        model.save("image_classifier.h5")
        print("Modell gespeichert: image_classifier.h5")

        # 3. TensorFlow Dataset erstellen
    def create_tf_dataset(self, images, labels, batch_size):
        """Erstellt ein TensorFlow-Dataset für das Training."""
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=len(images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

        # 4. Modell erstellen
    def create_model(self, input_shape, num_classes):
        """Erstellt ein einfaches CNN-Modell für die Bildklassifikation."""
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def update_current_label(self, label):
        """Aktualisiere das aktuelle Label."""
        self.current_label = label
        print(f"Aktuelles Label: {self.current_label}")

    def add_new_label(self):
        """Füge ein neues Label hinzu."""
        new_label, ok = QInputDialog.getText(self, "Neues Label", "Label eingeben:")
        if ok and new_label.strip():
            self.label_selector.addItem(new_label.strip())
            self.label_selector.setCurrentText(new_label.strip())
            print(f"Neues Label hinzugefügt: {new_label.strip()}")

    def save_label(self):
        """Speichere das Label für das aktuell ausgewählte Bild."""
        current_item = self.image_list.currentItem()
        if current_item is None or self.current_label is None:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie ein Bild und ein Label aus.")
            return

        file_path = current_item.data(Qt.ItemDataRole.UserRole)
        file_name = file_path.split("/")[-1]  # Nur der Dateiname
        self.labels[file_name] = self.current_label
        print(f"Label gespeichert: {file_name} -> {self.current_label}")
        QMessageBox.information(self, "Erfolg", f"Label '{self.current_label}' für das Bild gespeichert.")

    def export_labels(self):
        """Exportiere die Labels in eine JSON-Datei."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Labels exportieren", "", "JSON-Dateien (*.json)")
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.labels, f, indent=4)
            QMessageBox.information(self, "Erfolg", f"Labels erfolgreich exportiert: {file_path}")

    def loadStylesheet(self, filename):
        print('STYLE loading: ', filename)
        file = QFile(filename)
        file.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text)
        styleSheet = file.readAll()
        QApplication.instance().setStyleSheet(str(styleSheet, 'utf-8'))

    def create_thumbnail(self, file_path ,thumbnail_size):
        """Erstellt ein Vorschaubild für die Bildliste."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Bild konnte nicht geladen werden.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        aspect_ratio = w / h

        if aspect_ratio > 1:  # Breiteres Bild
            thumbnail_width = thumbnail_size
            thumbnail_height = int(thumbnail_size / aspect_ratio)
        else:  # Höheres Bild
            thumbnail_height = thumbnail_size
            thumbnail_width = int(thumbnail_size * aspect_ratio)
        # Resize das Bild für das Thumbnail
        thumbnail = cv2.resize(image, (thumbnail_width, thumbnail_height))
        qt_image = QImage(thumbnail.data, thumbnail.shape[1], thumbnail.shape[0], QImage.Format.Format_RGB888)
        return qt_image

    def resizeEvent(self, event):
        """Wird aufgerufen, wenn das Fenster seine Größe ändert."""
        self.adjust_thumbnail_size()
        super().resizeEvent(event)

    def adjust_thumbnail_size(self):
        """Passt die Größe der Thumbnails basierend auf der Fenstergröße an."""
        # Berechne die verfügbare Breite und Höhe für die Liste
        list_width = self.image_list.width() - 20  # Abziehen von Paddings und Rändern
        thumbnail_size = min(max(list_width, 50), 200)  # Dynamische Skalierung zwischen 50 und 200 Pixel

        # Passe die Größe der Thumbnails an
        self.image_list.setIconSize(QSize(thumbnail_size, thumbnail_size))

        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            file_path = item.data(Qt.ItemDataRole.UserRole)

            # Lade das Bild
            image = cv2.imread(file_path)
            if image is None:
                continue

            # Konvertiere das Bild in RGB und erzeuge ein Thumbnail
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            aspect_ratio = w / h

            # Proportionale Skalierung
            if aspect_ratio > 1:  # Breiteres Bild
                scaled_width = thumbnail_size
                scaled_height = int(thumbnail_size / aspect_ratio)
            else:  # Höheres Bild
                scaled_height = thumbnail_size
                scaled_width = int(thumbnail_size * aspect_ratio)

            # Resize das Bild für das Thumbnail
            thumbnail = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            thumbnail_qimage = QImage(thumbnail.data, thumbnail.shape[1], thumbnail.shape[0],
                                      QImage.Format.Format_RGB888)
            thumbnail_pixmap = QPixmap.fromImage(thumbnail_qimage)

            # Setze das aktualisierte Thumbnail und die Größe
            item.setIcon(QIcon(thumbnail_pixmap))
            item.setSizeHint(QSize(thumbnail_size, thumbnail_size))  # Platz im QListWidget reservieren

    def create_coco_structure(self):
        """Erstellt die Grundstruktur für COCO."""
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "region", "supercategory": "shape"}]
        }
    def load_images(self):
        """Mehrere Bilder laden und als dynamische Thumbnails in der Liste anzeigen."""
        file_paths, ndirectory = QFileDialog.getOpenFileNames(self, "Bilder auswählen", "", "Bilder (*.png *.jpg *.jpeg *.bmp)")

        if file_paths:
            self.image_paths = file_paths
            self.image_list.clear()  # Liste zurücksetzen
            self.images_path =  os.path.dirname(file_paths[0])


            for file_path in file_paths:
                # Erstelle ein Listenelement
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, file_path)  # Speichere den Pfad des Bildes
                item.setToolTip(file_path.split("/")[-1])  # Füge den Dateinamen als Tooltip hinzu
                self.image_list.addItem(item)

            # Passe die Thumbnails dynamisch an
            self.adjust_thumbnail_size()


    def load_selected_image(self, item):
        """Bild aus der Liste laden und anzeigen."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        file_name = file_path.split("/")[-1]  # Nur der Dateiname
        self.selected_Image_path= file_path

        # Bild laden
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            print(f"Fehler beim Laden des Bildes: {file_path}")
            return

        # BGR -> RGB umwandeln
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Bildgröße überprüfen und skalieren
        max_dim = 1000  # Maximale Breite oder Höhe
        h, w = self.original_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            self.original_image = cv2.resize(self.original_image, (int(w * scale), int(h * scale)),
                                             interpolation=cv2.INTER_AREA)
            print(f"Bild wurde skaliert: Neue Größe = {self.original_image.shape[1]}x{self.original_image.shape[0]}")

        # Maske initialisieren
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

        # Bild als QPixmap laden
        try:
            qimage = QImage(self.original_image.data, self.original_image.shape[1], self.original_image.shape[0],
                            QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"Fehler beim Konvertieren des Bildes: {e}")
            return


        # Bild zur COCO-Struktur hinzufügen
        self.coco_data["images"].append({
            "id": self.image_id_counter,
            "file_name": file_path.split("/")[-1],
            "height": h,
            "width": w
        })
        self.image_id_counter += 1
        # Bild zur Szene hinzufügen
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)
        self.undo_stack.clear()  # Stack zurücksetzen

        # Setze das gespeicherte Label, falls vorhanden
        if file_name in self.labels:
            self.label_selector.setCurrentText(self.labels[file_name])
        else:
            self.label_selector.setCurrentText("")

    def reset_view(self):
        """Setze Ansicht und Maske zurück."""
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.mask = np.zeros((h, w), dtype=np.uint8)
            self.load_selected_image(self.image_list.currentItem())
            self.undo_stack.clear()
            self.region_points.clear()

    def save_mask(self):
        """Maske speichern."""
        if self.mask is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Maske speichern", "", "Bilder (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            cv2.imwrite(file_path, self.mask)


        # COCO-Datei speichern
        coco_file_path = file_path.rsplit(".", 1)[0] + "_coco.json"  # Gleicher Pfad wie Maske, aber mit .json
        self.save_coco(coco_file_path)
        print(f"COCO-Datei gespeichert: {coco_file_path}")

    def redraw_mask_from_stack(self):
        """Erstelle die Maske basierend auf dem aktuellen Zustand des Undo-Stacks neu."""
        # Setze die Maske zurück
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.mask = np.zeros((h, w), dtype=np.uint8)

            # Zeichne alle verbleibenden Objekte aus dem Undo-Stack in die Maske
            for action in self.undo_stack:
                if action[0] == "Line":
                    # Zeichne das Liniensegment in die Maske
                    _, _, points = action
                    if len(points) > 1:
                        for i in range(len(points) - 1):
                            cv2.line(self.mask,(int(points[i].x()), int(points[i].y())),
                                (int(points[i + 1].x()), int(points[i + 1].y())),
                                color=255, thickness=self.line_width)

                elif action[0] == "Polygon":
                    # Zeichne das Polygon in die Maske
                    _, _, points = action
                    if len(points) > 2:
                        polygon = np.array([[int(p.x()), int(p.y())] for p in points], dtype=np.int32)
                        #cv2.fillPoly(self.mask, [polygon], 255)
                        # Zeichne die Kanten des Polygons in die Maske
                        # Aktualisiere die Anzeige mit der neuen Maske
                        self.update_mask_with_polygon(points)
                        #cv2.polylines(self.mask, [polygon], isClosed=True, color=255, thickness=self.line_width)
                elif action[0] == "Rectangle":
                    # Zeichne das Rechteck in die Maske
                    _, annotation_id, rect, rectangle_item, last_mask = action
                    x1, y1 = int(rect.x()), int(rect.y())
                    x2, y2 = int(rect.x() + rect.width()), int(rect.y() + rect.height())
                    cv2.rectangle(self.mask, (x1, y1), (x2, y2), color=255, thickness=-1)

                elif action[0] == "Circle":
                    # Zeichne den Kreis in die Maske
                    _, annotation_id, centre, radius, circle_item, last_mask = action
                    center = (int(centre.x()), int(centre.y()))
                    radius = int(radius)
                    cv2.circle(self.mask, center, radius, color=255, thickness=-1)


    def undo_last_action(self):
        """Rückgängig machen der letzten Aktion."""
        print("self.undo_stack Count Vor :", len(self.undo_stack))
        if self.undo_stack:
            # Letztes Objekt aus dem Undo-Stack entfernen
            last_action = self.undo_stack.pop()
            print("self.undo_stack Count Nach :", len(self.undo_stack))

            if last_action[0] == "Line":
                # Linie und Punkte zurücksetzen
                _, last_item, last_points = last_action
                if last_item:
                    self.scene.removeItem(last_item)
                self.region_points = last_points[:-1]  # Entferne den letzten Punkt
                self.redraw_mask_from_stack()
                self.update_original_image_with_maskT()
            elif last_action[0] == "Polygon":
                # Polygon und Punkte entfernen
                _, last_items, last_points = last_action
                for item in last_items:
                    self.scene.removeItem(item)
                self.region_points = []  # Zurücksetzen
                self.redraw_mask_from_stack()
                self.update_original_image_with_mask()
            elif last_action[0] == "Rectangle":
                # Rechteck entfernen
                _, annotation_id, rect, rectangle_item, last_mask = last_action
                self.scene.removeItem(rectangle_item)
                self.remove_annotation_from_coco(annotation_id)
                #self.mask = last_mask.copy()
                self.redraw_mask_from_stack()
                self.update_original_image_with_mask()
            elif last_action[0] == "Circle":
                # Kreis entfernen und Maske zurücksetzen
                _, annotation_id,centre, radius, circle_item, last_mask = last_action
                self.scene.removeItem(circle_item)
                self.remove_annotation_from_coco(annotation_id)
                #self.mask = last_mask.copy()
                self.redraw_mask_from_stack()
                self.update_original_image_with_mask()
                # Aktualisiere das Originalbild basierend auf der neuen Maske

            #self.redraw_mask_from_stack()
            # Aktualisiere das Originalbild basierend auf der neuen Maske
            #self.update_original_image_with_mask()
            #print("Letzte Aktion rückgängig gemacht.")

        #else:
        if (len(self.undo_stack) == 0):

            self.reset_view()
            self.update_original_image_with_mask()
        print("Nichts mehr zum Rückgängig machen.")

    def update_original_image_with_maskT(self):
        """Aktualisiert die Anzeige des Originalbilds basierend auf der Maske."""
        if self.original_image is not None and self.mask is not None:
            # Erstelle eine Kopie des Originalbilds und überlagere die Maske
            overlay = self.original_image.copy()
            overlay[self.mask == 255] = [255, 0, 0]  # Markiere maskierte Bereiche in Rot

            # Konvertiere das aktualisierte Bild in QPixmap und zeige es an
            h, w = overlay.shape[:2]
            qimage = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # Entferne das alte Bild aus der Szene
            if self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)

            # Füge das aktualisierte Bild in die Szene ein
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            print("Originalbild mit aktualisierter Maske gerendert.")

    def update_original_image_with_mask(self):
        """Aktualisiert das Originalbild mit einer transparenten roten Maske und durchgestrichenen Linien nur innerhalb der Maske."""
        if self.original_image is not None and self.mask is not None:
            # Kopiere das Originalbild
            overlay = self.original_image.copy()

            # Erstelle eine transparente rote Maske (im BGR-Format)
            transparent_red = np.zeros_like(overlay, dtype=np.uint8)
            transparent_red[:, :, 2] = 255  # Rot-Kanal (dritter Kanal in BGR)

            # Transparenz mischen (nur innerhalb der Maske)
            alpha = 0.2  # Transparenzlevel (0 = komplett transparent, 1 = komplett sichtbar)
            mask_indices = self.mask == 255
            overlay[mask_indices] = (
                        (1 - alpha) * overlay[mask_indices] + alpha * transparent_red[mask_indices]).astype(np.uint8)

            # Füge durchgestrichene Linien NUR innerhalb der Maske hinzu
            mask_with_lines = overlay.copy()  # Kopie für die Linien
            line_color = (0, 0, 255)  # Weiß für durchgestrichene Linien
            line_thickness = 1  # Dicke der Linien
            line_spacing = 50  # Abstand zwischen den Linien

            # Zeichne die Linien innerhalb der Maske
            for y in range(0, self.mask.shape[0], line_spacing):
                for x in range(0, self.mask.shape[1], line_spacing):
                    if self.mask[y, x] == 255:  # Nur innerhalb der Maske zeichnen
                        # Diagonale Linien innerhalb der Maske
                        if x + line_spacing < self.mask.shape[1] and y + line_spacing < self.mask.shape[0]:
                            cv2.line(mask_with_lines, (x, y), (x + line_spacing, y + line_spacing), line_color,
                                     thickness=line_thickness)
                        if x - line_spacing >= 0 and y + line_spacing < self.mask.shape[0]:
                            cv2.line(mask_with_lines, (x, y), (x - line_spacing, y + line_spacing), line_color,
                                     thickness=line_thickness)

            # Kombiniere das Originalbild und die Maske mit Linien
            overlay[mask_indices] = mask_with_lines[mask_indices]

            # Konvertiere das aktualisierte Bild in QPixmap und zeige es an
            h, w = overlay.shape[:2]
            qimage = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)

            # Entferne das alte Bild aus der Szene
            if self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)

            # Füge das aktualisierte Bild in die Szene ein
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            print("Originalbild mit transparenter Maske und durchgestrichenen Linien innerhalb der Maske aktualisiert.")



    def invert_and_show_mask(self):
        """Invertiert die Maske und zeigt sie auf dem Originalbild."""
        if self.mask is not None and self.original_image is not None:
            # Maske invertieren
            inverted_mask = cv2.bitwise_not(self.mask)
            self.mask = inverted_mask
            # Speichere das gezeichnete Objekt und die aktuelle Maske im Stack
            self.undo_stack.append((self.temp_item, self.mask.copy()))
            print("Aktion gespeichert.")
            # Maske auf Originalbild anwenden (Transparenz 50%)
            overlay = self.original_image.copy()
            overlay[inverted_mask == 255] = [255, 0, 0]  # Rot für invertierte Bereiche

            # Zeige das Ergebnis in der Szene
            h, w = overlay.shape[:2]
            qimage = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # Entferne das alte Bild
            if self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)

            # Zeige das neue Bild
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            print("Maske invertiert und auf Originalbild angezeigt.")

    def update_drawing_mode(self, mode):
        """Aktualisiere den Zeichenmodus."""
        self.drawing_mode = mode
        print(f"Zeichenmodus: {self.drawing_mode}")

    def update_line_width(self, value):
        """Aktualisiert die Breite der Linien."""
        self.line_width = value
        print(f"Linienbreite aktualisiert: {self.line_width}")

    def apply_eraser(self, position):
        """Radierer: Lösche alle gezeichneten Inhalte im angegebenen Bereich."""
        if self.mask is not None:
            # Radierbereich definieren
            eraser_radius = self.line_width
            center = (int(position.x()), int(position.y()))

            # Entferne den Radierbereich in der Maske
            cv2.circle(self.mask, center, eraser_radius, color=0, thickness=-1)

            # Aktualisiere die Anzeige mit der neuen Maske
            self.update_original_image_with_mask()
    def on_scene_mouse_press(self, event):
        """Zeichnen der Region starten."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.scenePos()
            self.last_point = self.start_point
            self.drawing = True

            # Drag-Modus deaktivieren
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag)

            if self.drawing_mode == "Eraser":
                # Lösche sofort den Bereich an der aktuellen Mausposition
                self.apply_eraser(event.scenePos())

    def on_scene_mouse_move(self, event):
        """Während des Zeichnens."""
        if self.drawing and self.start_point:
            scene_pos = event.scenePos()

            if self.drawing_mode == "Eraser":
                # Radierer: Lösche Pixel in der Maske
                self.apply_eraser(scene_pos)
            # Temporäres Element für Rechteck und Kreis
            elif (self.drawing_mode == "Rectangle"
                    or self.drawing_mode == "Circle"):
                if self.temp_item:
                    self.scene.removeItem(self.temp_item)

                if self.drawing_mode == "Rectangle":
                    rect = QRectF(self.start_point, scene_pos)
                    self.temp_item = self.scene.addRect(rect, QPen(Qt.GlobalColor.blue, self.line_width))
                elif self.drawing_mode == "Circle":
                    center = self.start_point
                    radius = ((scene_pos.x() - self.start_point.x())**2 +
                              (scene_pos.y() - self.start_point.y())**2) ** 0.5
                    self.temp_item = self.scene.addEllipse(center.x() - radius, center.y() - radius,
                                                           2 * radius, 2 * radius,
                                                           QPen(Qt.GlobalColor.green, self.line_width))
            elif self.drawing_mode == "Line":
                # Zeichnen der Linie
                pen = QPen(Qt.GlobalColor.red, self.line_width)
                self.region_points.append(event.scenePos())
                self.temp_item = self.scene.addLine(self.last_point.x(), self.last_point.y(),
                                   scene_pos.x(), scene_pos.y(), pen)
                # Maske aktualisieren
                cv2.line(self.mask,
                         (int(self.last_point.x()), int(self.last_point.y())),
                         (int(scene_pos.x()), int(scene_pos.y())),
                         color=255, thickness=self.line_width)

                self.last_point = scene_pos

    def on_scene_mouse_release(self, event):
        """Zeichnen der Region beenden."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.temp_item:
                self.scene.removeItem(self.temp_item)
                self.temp_item = None

            scene_pos = event.scenePos()

            if self.drawing_mode == "Rectangle":
                # Rechteck zeichnen
                rect = QRectF(self.start_point, scene_pos)
                rectangle_item = self.scene.addRect(rect, QPen(Qt.GlobalColor.blue, self.line_width))

                # Rechteck zur Maske hinzufügen
                x1, y1 = int(self.start_point.x()), int(self.start_point.y())
                x2, y2 = int(scene_pos.x()), int(scene_pos.y())
                cv2.rectangle(self.mask, (x1, y1), (x2, y2), color=255, thickness=-1)

                # Rechteck zur COCO-Struktur hinzufügen
                self.add_rectangle_to_coco(rect, rectangle_item, self.mask.copy())
                # Aktualisiere die Anzeige mit der neuen Maske
                self.update_original_image_with_mask()

            elif self.drawing_mode == "Circle":
                # Kreis zeichnen
                radius = ((scene_pos.x() - self.start_point.x()) ** 2 + (
                            scene_pos.y() - self.start_point.y()) ** 2) ** 0.5
                circle_item = self.scene.addEllipse(
                    self.start_point.x() - radius, self.start_point.y() - radius,
                    2 * radius, 2 * radius, QPen(Qt.GlobalColor.green, self.line_width)
                )
                # Kreis zur Maske hinzufügen
                center = (int(self.start_point.x()), int(self.start_point.y()))
                cv2.circle(self.mask, center, int(radius), color=255, thickness=-1)
                # Kreis zur COCO-Struktur hinzufügen
                self.add_circle_to_coco(self.start_point, radius, circle_item, self.mask.copy())
                # Aktualisiere die Anzeige mit der neuen Maske
                self.update_original_image_with_mask()

            elif self.drawing_mode == "Line":
                # Benutzer hat das Zeichnen beendet (z. B. durch Rechtsklick)
                if len(self.region_points) > 1:
                    # Prüfe, ob die Linie geschlossen ist (Abstand zwischen erstem und letztem Punkt)
                    first_point = self.region_points[0]
                    last_point = self.region_points[-1]
                    distance = ((first_point.x() - last_point.x()) ** 2 + (
                                first_point.y() - last_point.y()) ** 2) ** 0.5

                    if distance < 10:  # Schwelle für geschlossene Region
                        # Linie ist geschlossen -> Als Polygon behandeln
                        closing_line = self.scene.addLine(
                            last_point.x(), last_point.y(),
                            first_point.x(), first_point.y(),
                            QPen(Qt.GlobalColor.red, 2)
                        )
                        # Speichere das Polygon (Linien und Punkte) im Undo-Stack
                        self.undo_stack.append(("Polygon", [closing_line], self.region_points.copy()))
                        self.add_closed_region_to_coco(self.region_points)
                        # Aktualisiere die Anzeige mit der neuen Maske
                        self.update_mask_with_polygon(self.region_points)
                        self.update_original_image_with_mask()
                        print("Geschlossene Linie erkannt und als Polygon gespeichert.")
                    else:
                        self.undo_stack.append(("Line", None, self.region_points.copy()))  # Speichere nur Punkte
                        # Linie ist nicht geschlossen -> Als offene Linie speichern
                        self.add_open_line_to_coco(self.region_points)
                        print("Offene Linie erkannt und gespeichert.")

                    # Punkte zurücksetzen
                    self.region_points = []

            if self.drawing_mode == "Eraser":
                print("Radierer-Aktion abgeschlossen.")

            self.drawing = False
            self.start_point = None
            self.last_point = None

            # Drag-Modus wieder aktivieren
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def update_mask_with_polygon(self, points):
        """Zeichne die geschlossene Region in die Maske."""
        if self.mask is not None:
            polygon = np.array([[int(p.x()), int(p.y())] for p in points], dtype=np.int32)
            cv2.fillPoly(self.mask, [polygon], 255)  # Fülle das Polygon in der Maske aus
            print("Maske mit geschlossener Region aktualisiert.")

    def add_rectangle_to_coco(self, rect, rectangle_Item, mask):
        """Füge ein Rechteck zur COCO-Struktur hinzu."""
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        bbox = [x, y, w, h]

        annotation = {
            "id": self.annotation_id_counter,
            "image_id": self.image_id_counter - 1,
            "segmentation": segmentation,
            "bbox": bbox,
            "category_id": 1,
            "iscrowd": 0
        }
        self.coco_data["annotations"].append(annotation)

        # Speichere die Annotation-ID im Undo-Stack
        self.undo_stack.append(("Rectangle", annotation["id"], rect, rectangle_Item, mask))
        self.annotation_id_counter += 1

    def add_circle_to_coco(self, center, radius, circle_item, mask):
        """Füge einen Kreis als Polygon zur COCO-Struktur hinzu."""
        num_points = 36
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center.x() + radius * np.cos(angle)
            y = center.y() + radius * np.sin(angle)
            points.append(x)
            points.append(y)
        bbox = [center.x() - radius, center.y() - radius, 2 * radius, 2 * radius]

        annotation = {
            "id": self.annotation_id_counter,
            "image_id": self.image_id_counter - 1,
            "segmentation": [points],
            "bbox": bbox,
            "category_id": 1,
            "iscrowd": 0
        }
        self.coco_data["annotations"].append(annotation)

        # Speichere Kreis und Maske im Undo-Stack
        self.undo_stack.append(("Circle", annotation["id"], center, radius, circle_item, mask))
        print("Kreis gespeichert und zur Maske hinzugefügt.")
        self.annotation_id_counter += 1

    def add_closed_region_to_coco(self, points):
        """Füge eine geschlossene Region zur COCO-Struktur hinzu."""
        # Konvertiere Punkte in flache Liste für COCO
        segmentation = [p.x() for p in points] + [p.y() for p in points]

        # Bounding-Box berechnen
        polygon = np.array([[p.x(), p.y()] for p in points], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(polygon)

        # Zur COCO-Struktur hinzufügen
        self.coco_data["annotations"].append({
            "id": self.annotation_id_counter,
            "image_id": self.image_id_counter - 1,
            "segmentation": [segmentation],  # Polygonpunkte als flache Liste
            "bbox": [x, y, w, h],
            "category_id": 1,
            "iscrowd": 0
        })
        self.annotation_id_counter += 1

    def add_open_line_to_coco(self, points):
        """Füge eine offene Linie zur COCO-Struktur hinzu."""
        # Konvertiere Punkte in flache Liste für COCO
        segmentation = [p.x() for p in points] + [p.y() for p in points]

        # Bounding-Box berechnen
        x_coords = [p.x() for p in points]
        y_coords = [p.y() for p in points]
        x, y, w, h = min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)

        # Zur COCO-Struktur hinzufügen
        self.coco_data["annotations"].append({
            "id": self.annotation_id_counter,
            "image_id": self.image_id_counter - 1,
            "segmentation": [segmentation],  # Linienpunkte als flache Liste
            "bbox": [x, y, w, h],
            "category_id": 2,  # Neue Kategorie für offene Linien
            "iscrowd": 0
        })
        self.annotation_id_counter += 1

    def save_coco(self, file_path):
        """Speichere die COCO-Datenstruktur als JSON."""
        # COCO-Datenstruktur als JSON speichern
        with open(file_path, "w") as f:
            json.dump(self.coco_data, f, indent=4)
        print(f"COCO-Datei gespeichert: {file_path}")

    def remove_annotation_from_coco(self, annotation_id):
        """Entferne eine Annotation aus der COCO-Datenstruktur."""
        self.coco_data["annotations"] = [
            annotation for annotation in self.coco_data["annotations"] if annotation["id"] != annotation_id
        ]
        print(f"Annotation mit ID {annotation_id} entfernt.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Wende das Stylesheet an
    app.setStyleSheet(dark_stylesheet)

    viewer = ImageViewerApp()
    viewer.show()
    sys.exit(app.exec())
