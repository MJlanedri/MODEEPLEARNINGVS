import sys
from PyQt6.QtCore import QFile, QIODevice
import cv2
import json
import os
import numpy as np
from trainmulticlassmodel import train_multiclass_model

from CustomGraphicsScene import CustomGraphicsScene
from CustomTitleBar import CustomTitleBar
from GraphicsView import GraphicsView
from utils import loadStylesheet
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
    QFileDialog, QListWidget, QGraphicsView, QGraphicsPixmapItem, QSlider, QLabel, QComboBox,
    QListWidgetItem, QMessageBox, QInputDialog, QGroupBox, QGridLayout, QAbstractItemView
)
from PyQt6.QtGui import QPixmap, QImage, QPen, QIcon
from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, QFile
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.models import load_model
from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten,
    Dense, GlobalAveragePooling2D
)
from keras.optimizers import Adam, SGD
from keras.initializers import HeNormal
from keras.utils import to_categorical
from skimage import transform
from keras.activations import relu


class ImageViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()

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
        self.setCentralWidget(self.central_widget)

        # Pfade und Parameter
        self.json_file_path = "./Labels/labels.json" # Pfad zur JSON-Datei mit Labels
        self.input_shape = (64, 64, 3)
        self.batch_size = 128  # Batch-Größe
        self.num_epochs = 100  # Anzahl der Epoche
        self.output_Dataset_file_path = "./Datasets"
        self.test_size = 0.2
        self.validation_split = 0.5  # Anteil der Validierungsdaten aus dem temporären Datensatz


        # Hauptvariablen
        self.Modell=None
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
        self.num_classes = 2
        self.output_predictions_path = "./Predictions./predictions.json"
        self.label_mapping_path = "./Labeles_Mapping"
        self.output_Model_path="./Models"

        # UI-Komponenten
        self.Main_layout = QHBoxLayout()

        # Linke Bildliste
        self.image_list = QListWidget()
        self.image_list.setFixedWidth(300)
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
        self.reset_button = QPushButton("Region Zurücksetzen")
        self.undo_button = QPushButton("Region Rückgängig")  # Neue Schaltfläche für Rückgängig
        self.save_button = QPushButton("Region speichern")
        self.invert_button = QPushButton("Region umkehren")  # Neue Schaltfläche zum Umkehren der Maske
        self.label_save_button = QPushButton("Label speichern")
        self.label_export_button = QPushButton("Labels exportieren")
        self.import_Model_button = QPushButton("Model laden")
        self.train_dataset = QPushButton("Modell trainieren")
        self.test_dataset = QPushButton("Modell testen")

        # 1. Bildoptionen
        self.image_group = QGroupBox("Bild")
        self.image_layout = QVBoxLayout()
        self.image_layout.addWidget(self.load_button)
        self.image_group.setLayout(self.image_layout)
        self.image_grid_layout = QGridLayout()
        self.image_grid_layout.addWidget(self.image_group, 0, 0)  # Bildoptionen links oben
        self.button_layout.addLayout(self.image_grid_layout)

        # 2.Modusobtion
        self.mode_group = QGroupBox("Supervisedregion")
        # Dropdown für Modusauswahl
        self.All_mode_layout = QVBoxLayout()
        self.mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Modus:")
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Line", "Rectangle", "Circle", "Eraser"])
        self.mode_selector.currentTextChanged.connect(self.update_drawing_mode)
        self.mode_layout.addWidget(self.mode_label)
        self.mode_layout.addWidget(self.mode_selector)
        self.All_mode_layout.addLayout(self.mode_layout)

        self.All_mode_layout.addWidget(self.undo_button)
        self.All_mode_layout.addWidget(self.invert_button)
        self.All_mode_layout.addWidget(self.reset_button)
        self.All_mode_layout.addWidget(self.save_button)
        self.mode_group.setLayout(self.All_mode_layout)
        self.mode_grid_layout = QGridLayout()
        self.mode_grid_layout.addWidget(self.mode_group, 0, 0)  # Bildoptionen links oben
        self.button_layout.addLayout(self.mode_grid_layout)


        # 2. Labelobtion
        self.Label_group = QGroupBox("Klassifikator")
        self.LabelAll_layout = QVBoxLayout()
        # Label-Management
        self.label_layout = QHBoxLayout()
        self.label_selector = QComboBox()
        self.label_selector.setEditable(True)  # Benutzer kann neue Labels hinzufügen
        self.label_selector.addItems(["OK", "NOK"])  # Beispiel-Labels
        self.label_selector.currentTextChanged.connect(self.update_current_label)
        self.label_layout.addWidget(QLabel("Label:"))
        self.label_layout.addWidget(self.label_selector)
        self.label_add_button = QPushButton("Label hinzufügen")
        self.label_add_button.clicked.connect(self.add_new_label)
        self.LabelAll_layout.addLayout(self.label_layout)

        self.label_action_combo = QComboBox()  # Dropdown-Menü für Labeling-Optionen
        self.label_action_combo.addItems([
            "Ein ausgewähltes Bild labeln",
            "Mehrere ausgewählte Bilder labeln",
            "Alle Bilder labeln"
        ])

        #self.label_layout.addWidget(self.label_add_button)
        self.label_Test_button = QPushButton("Labeln")
        self.save_Test_button = QPushButton("Labels speichern")

        #self.LabelAll_layout.addWidget(self.label_add_button)
        #self.LabelAll_layout.addWidget(self.label_save_button)
        #self.LabelAll_layout.addWidget(self.label_export_button)
        self.LabelAll_layout.addWidget(self.label_action_combo)
        self.LabelAll_layout.addWidget(self.label_Test_button)
        self.LabelAll_layout.addWidget(self.save_Test_button)
        self.Label_group.setLayout(self.LabelAll_layout)

         # Verbindungen

        self.label_Test_button.clicked.connect(self.perform_Test_label_action)
        self.save_Test_button.clicked.connect(self.save_Test_labels)


        self.label_grid_layout = QGridLayout()
        self.label_grid_layout.addWidget(self.Label_group, 0, 0)  # Bildoptionen links oben
        self.button_layout.addLayout(self.label_grid_layout)

        # 4. trainobtion
        self.train_group = QGroupBox("Modell")
        self.train_layout = QVBoxLayout()
        self.train_layout.addWidget(self.train_dataset)
        self.train_layout.addWidget(self.import_Model_button)
        self.train_layout.addWidget(self.test_dataset)
        self.train_group.setLayout(self.train_layout)
        self.train_grid_layout = QGridLayout()
        self.train_grid_layout.addWidget(self.train_group, 0, 0)  # Bildoptionen links oben
        self.button_layout.addLayout(self.train_grid_layout)




        self.right_layout.addLayout(self.button_layout)


        #self.right_layout.addLayout(self.label_layout)





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
        self.train_dataset.clicked.connect(self.creat_datset_and_train)
        self.import_Model_button.clicked.connect(self.model_laden)
        self.test_dataset.clicked.connect(self.test_NModel)

    def perform_Test_label_action(self):
        """Führt die Labeling-Aktion basierend auf der Dropdown-Auswahl aus."""
        selected_action = self.label_action_combo.currentText()

        if selected_action == "Ein ausgewähltes Bild labeln":
            self.label_single_image()
        elif selected_action == "Mehrere ausgewählte Bilder labeln":
            self.label_selected_images()
        elif selected_action == "Alle Bilder labeln":
            self.label_all_images()
        else:
            QMessageBox.warning(self, "Fehler", "Ungültige Aktion ausgewählt.")


    def label_single_image(self):
        """Labelt ein einzelnes ausgewähltes Bild."""
        selected_items = self.image_list.selectedItems()
        if len(selected_items) != 1:
            QMessageBox.warning(self, "Fehler", "Bitte wählen Sie genau ein Bild aus.")
            return
        if self.current_label is None:
           # Benutzer nach einem Label fragen
            label, ok = QInputDialog.getText(self, "Label festlegen", "Geben Sie ein Label für das ausgewählte Bild ein:")
            if not ok or not label:
                return
        else:
            label = self.current_label



        file_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self.labels[os.path.basename(file_path)] = label

        QMessageBox.information(self, "Erfolg", f"Das Bild wurde mit dem Label '{label}' versehen.")

    def label_selected_images(self):
        """Labelt mehrere ausgewählte Bilder."""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Fehler", "Keine Bilder ausgewählt.")
            return


        if self.current_label is None:
           # Benutzer nach einem Label fragen
            label, ok = QInputDialog.getText(self, "Label festlegen", "Geben Sie ein Label für die ausgewählten Bilder ein:")
            if not ok or not label:
                return
        else:
            label = self.current_label
        # Benutzer nach einem Label fragen



        for item in selected_items:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            self.labels[os.path.basename(file_path)] = label

        QMessageBox.information(self, "Erfolg", f"{len(selected_items)} Bilder wurden mit dem Label '{label}' versehen.")

    def label_all_images(self):
        """Labelt alle geladenen Bilder."""
        if not self.image_paths:
            QMessageBox.warning(self, "Fehler", "Es wurden keine Bilder geladen.")
            return


        if self.current_label is None:
           # Benutzer nach einem Label fragen
            label, ok = QInputDialog.getText(self, "Label festlegen", "Geben Sie ein Label für alle Bilder ein:")
            if not ok or not label:
                return
        else:
            label = self.current_label


        for file_path in self.image_paths:
            self.labels[os.path.basename(file_path)] = label

        QMessageBox.information(self, "Erfolg", f"Alle Bilder wurden mit dem Label '{label}' versehen.")


    def save_Test_labels(self):
        """Labels in einer JSON-Datei speichern."""
        if not self.labels:
            QMessageBox.warning(self, "Fehler", "Es wurden keine Labels erstellt.")
            return

        #file_path, _ = QFileDialog.getSaveFileName(self, "Labels speichern", "labels.json", "JSON-Dateien (*.json)")
        file_path = self.json_file_path
        if not file_path:
            return

        # Speichere nur die Bildnamen und Labels
        with open(file_path, "w") as f:
            json.dump(self.labels, f, indent=4)

        QMessageBox.information(self, "Erfolg", f"Labels wurden in '{file_path}' gespeichert.")



    def model_laden(self):
        # Modell laden
        model_path = os.path.join(self.output_Model_path, "image_classifier.h5")
        self.Modell = load_model(model_path)
        QMessageBox.information(self, "Erfolg", "Modell erfolgreich geladen!")


    def get_all_json_files(self, directory):
        """Gibt eine Liste aller JSON-Dateien in einem Verzeichnis zurück."""
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        return json_files

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
        import os

    def test_NModel(self):
        # Teste das Modell
        predictions = self.test_Model(
        test_image_path=self.selected_Image_path,
        img_shape=self.input_shape,
        label_mapping_path=self.label_mapping_path,
        output_predictions_path=self.output_predictions_path,
        )
        # Ausgabe der Vorhersagen
        for prediction in predictions:
            print(prediction)

    def test_Model(self,
        test_image_path,
        img_shape,
        label_mapping_path=None,
        output_predictions_path=None,
    ):
        """
        Testet ein trainiertes Modell mit neuen Testbildern.

        Args:
            model_path (str): Pfad zum gespeicherten Modell (.h5-Datei).
            test_images_dir (str): Verzeichnis mit Testbildern.
            img_shape (tuple): Zielgröße der Bilder (Höhe, Breite, Kanäle).
            label_mapping_path (str, optional): Pfad zur Label-Mapping-Datei (JSON).
            output_predictions_path (str, optional): Pfad zum Speichern der Vorhersagen (JSON).

        Returns:
            list: Liste von Vorhersagen mit Bildnamen und vorhergesagten Klassen.
        """
        # Lade das Modell

        #model_path = os.path.join(self.output_Model_path, "image_classifier.h5")
        #.Modell = load_model(model_path)
        #print("Modell geladen.")

        # Lade die Label-Mapping-Datei, falls angegeben
        label_mapping = None
        inverse_label_mapping = None
        mapping_path = os.path.join(label_mapping_path, "label_mapping.json")
        if mapping_path:
            import json
            with open(mapping_path, "r") as f:
                label_mapping = json.load(f)
                # Erstelle eine Umkehrung der Mapping-Tabelle (Zahl -> Name)
                inverse_label_mapping = {v: k for k, v in label_mapping.items()}
            print(f"Label-Mapping aus '{mapping_path}' geladen.")

        predictions = []

        try:
            # Lade das Bild
            img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
            img_name = os.path.basename(test_image_path)
            if img is None:
                print(f"Bild '{img_name}' konnte nicht geladen werden.")
                return

            # Konvertiere zu RGB und skaliere auf die Zielgröße
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform.resize(img, img_shape, preserve_range=True) / 255.0

            # Füge eine zusätzliche Dimension für die Batchgröße hinzu
            img = np.expand_dims(img, axis=0)

            # Mache eine Vorhersage
            prediction = self.Modell.predict(img)
            predicted_class = np.argmax(prediction)  # Index der höchsten Wahrscheinlichkeit
            confidence = np.max(prediction)  # Wahrscheinlichkeit der vorhergesagten Klasse

            # Übersetze in den Klassennamen, falls Mapping verfügbar ist
            if inverse_label_mapping:
                predicted_class_name = inverse_label_mapping[predicted_class]
            else:
                predicted_class_name = str(predicted_class)

            predictions.append({
                "image": img_name,
                "predicted_class": predicted_class_name,
                "confidence": float(confidence),
            })

            print(f"Bild '{img_name}': Vorhergesagte Klasse = {predicted_class_name}, "
                f"Confidence = {confidence:.2f}")
        except Exception as e:
            print(f"Fehler beim Verarbeiten von '{img_name}': {e}")

        # Speichere die Vorhersagen, falls ein Pfad angegeben ist
        if output_predictions_path:
            with open(output_predictions_path, "w") as f:
                import json
                json.dump(predictions, f, indent=4)
            print(f"Vorhersagen gespeichert unter '{output_predictions_path}'.")

        # Zeige das vorhergesagte Label als Overlay
        self.overlay_image = self.add_overlay(self.original_image, predicted_class_name)

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

        return predictions



    def test_Modell_(self):

        # Label-Mapping
        label_map = {0: "Katze", 1: "Schmitterling"}  # Passe dies an dein Mapping an
        # Vorhersage
        image_path = self.selected_Image_path
        image = self.preprocess_image(image_path, (self.image_size_One,  self.image_size_One))
        prediction = self.Modell.predict(image)
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
    def creat_datset_and_train(self):
        # Modell trainieren
        history, model = train_multiclass_model(
        json_file=self.json_file_path,
        image_dir=self.images_path,
        output_dataset_dir=self.output_Dataset_file_path,
        output_model_dir=self.output_Model_path,
        outputs_mapping_labeles=self.label_mapping_path,
        input_shape=self.input_shape,
        num_classes=self.num_classes,
        build_model_fn=self.build_model,
        optimizer=Adam,
        learning_rate=0.001,
        filter_block1=32,
        kernel_size_block1=3,
        filter_block2=64,
        kernel_size_block2=3,
        filter_block3=128,
        kernel_size_block3=3,
        filter_block4=256,
        kernel_size_block4=3,
        dense_layer_size=128,
        kernel_initializer="glorot_uniform",
        activation_cls=relu,
        dropout_rate=0.5,
        use_batch_normalization=True,
        use_dense=True,
        use_global_pooling=False,
        batch_size=self.batch_size,
        epochs=self.num_epochs,
        )
        model_path = os.path.join(self.output_Model_path, "image_classifier.h5")
        self.Modell = load_model(model_path)

        # 3. TensorFlow Dataset erstellen
    # TensorFlow-Datensätze
    def create_tf_dataset(self, images, labels, batch_size):
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


    def build_model(self,
        optimizer,
        learning_rate: float,
        filter_block1: int,
        kernel_size_block1: int,
        filter_block2: int,
        kernel_size_block2: int,
        filter_block3: int,
        kernel_size_block3: int,
        filter_block4: int,
        kernel_size_block4: int,
        dense_layer_size: int,
        kernel_initializer,
        activation_cls,
        dropout_rate: float,
        use_batch_normalization: bool,
        use_dense: bool,
        use_global_pooling: bool,
        num_classes: int
    ) -> Model:
        input_img = Input(self.input_shape)

        x = Conv2D(
            filters=filter_block1,
            kernel_size=kernel_size_block1,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(input_img)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = activation_cls(x)
        x = Conv2D(
            filters=filter_block1,
            kernel_size=kernel_size_block1,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        if dropout_rate:
            x = Dropout(rate=dropout_rate)(x)
        x = activation_cls(x)
        x = MaxPool2D()(x)

        x = Conv2D(
            filters=filter_block2,
            kernel_size=kernel_size_block2,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = activation_cls(x)
        x = Conv2D(
            filters=filter_block2,
            kernel_size=kernel_size_block2,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        if dropout_rate:
            x = Dropout(rate=dropout_rate)(x)
        x = activation_cls(x)
        x = MaxPool2D()(x)

        x = Conv2D(
            filters=filter_block3,
            kernel_size=kernel_size_block3,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = activation_cls(x)
        x = Conv2D(
            filters=filter_block3,
            kernel_size=kernel_size_block3,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        if dropout_rate:
            x = Dropout(rate=dropout_rate)(x)
        x = activation_cls(x)
        x = MaxPool2D()(x)

        x = Conv2D(
            filters=filter_block4,
            kernel_size=kernel_size_block4,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        x = activation_cls(x)
        x = Conv2D(
            filters=filter_block4,
            kernel_size=kernel_size_block4,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if use_batch_normalization:
            x = BatchNormalization()(x)
        if dropout_rate:
            x = Dropout(rate=dropout_rate)(x)
        x = activation_cls(x)
        x = MaxPool2D()(x)

        if use_global_pooling:
            x = GlobalAveragePooling2D()(x)
        else:
            x = Flatten()(x)
        if use_dense:
            x = Dense(
                units=dense_layer_size,
                kernel_initializer=kernel_initializer,
            )(x)
            if use_batch_normalization:
                x = BatchNormalization()(x)
            x = activation_cls(x)
        x = Dense(
            units=num_classes,  # Beispiel mit 10 Klassen
            kernel_initializer=kernel_initializer,
        )(x)
        y_pred = Activation("softmax")(x)

        model = Model(
            inputs=[input_img],
            outputs=[y_pred],
        )

        opt = optimizer(learning_rate=learning_rate)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )
        model.summary()

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
        #file_path, _ = QFileDialog.getSaveFileName(self, "Labels exportieren", "", "JSON-Dateien (*.json)")
        file_path = self.json_file_path
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
    #app.setStyleSheet(dark_stylesheet)

    loadStylesheet("./qss./dark_stylesheet.qss")

    viewer = ImageViewerApp()
    viewer.show()
    sys.exit(app.exec())
