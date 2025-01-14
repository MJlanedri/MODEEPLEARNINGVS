import os
import numpy as np
import cv2
from skimage import transform
import json


class BinaryDatasetCreator:
    def __init__(self, json_file, image_dir, img_shape=(64, 64, 3)):
        """
        Initialisiert den Dataset-Creator.

        Args:
            json_file (str): Pfad zur JSON-Datei mit gelabelten Bildern.
            image_dir (str): Verzeichnis, in dem sich die Bilder befinden.
            img_shape (tuple): Zielgröße der Bilder (Höhe, Breite, Kanäle).
        """
        self.json_file = json_file
        self.image_dir = image_dir
        self.img_shape = img_shape
        self.data = self.load_json()

        # Dynamische Mapping-Tabelle für Labels
        self.label_mapping = {}

    def load_json(self):
        """Lädt die JSON-Datei."""
        if not os.path.exists(self.json_file):
            raise FileNotFoundError(f"JSON-Datei '{self.json_file}' nicht gefunden!")
        with open(self.json_file, "r") as f:
            return json.load(f)

    def update_label_mapping(self, label_name):
        """
        Fügt ein neues Label zur Mapping-Tabelle hinzu, wenn es nicht existiert.

        Args:
            label_name (str): Der Name des Labels (z. B. 'cat').
        """
        if label_name not in self.label_mapping:
            new_id = len(self.label_mapping)  # Neue ID ist die nächste freie Zahl
            self.label_mapping[label_name] = new_id
            print(f"Neues Label hinzugefügt: '{label_name}' -> {new_id}")

    def process_images(self):
        """
        Verarbeitet die Bilder basierend auf den Labels in der JSON-Datei.

        Returns:
            tuple: Arrays für die Bilder (x) und Labels (y).
        """
        images = []
        labels = []

        for file_name, label_name in self.data.items():
            try:
                # Kombiniere den Bildnamen mit dem Verzeichnis
                img_file_path = os.path.join(self.image_dir, file_name)

                # Füge neues Label zur Mapping-Tabelle hinzu, falls erforderlich
                self.update_label_mapping(label_name)

                # Lade das Bild
                img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Bild '{img_file_path}' konnte nicht geladen werden.")
                    continue

                # Konvertiere zu RGB und skaliere auf die Zielgröße
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform.resize(img, self.img_shape, preserve_range=True) / 255.0

                # Labels in Zahlen umwandeln
                numeric_label = self.label_mapping[label_name]
                images.append(img)
                labels.append(numeric_label)
            except Exception as e:
                print(f"Fehler beim Verarbeiten von '{file_name}': {e}")

        # Konvertiere zu NumPy-Arrays
        x = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        return x, y

    def save_binary_files(self, output_Label_mapping_dir):
        """
        Speichert die Bilder und Labels als .npy-Dateien.

        Args:
            output_dir (str): Zielverzeichnis für die .npy-Dateien.
        """
        if not os.path.exists(output_Label_mapping_dir):
            os.makedirs(output_Label_mapping_dir)

        x, y = self.process_images()

        x_filepath = os.path.join(output_Label_mapping_dir, "x.npy")
        y_filepath = os.path.join(output_Label_mapping_dir, "y.npy")
        mapping_filepath = os.path.join(output_Label_mapping_dir, "label_mapping.json")

        # Speichere die verarbeiteten Daten
        np.save(x_filepath, x)
        np.save(y_filepath, y)

        # Speichere die Label-Mapping-Tabelle als JSON
        with open(mapping_filepath, "w") as f:
            json.dump(self.label_mapping, f, indent=4)

        print(f"Binärdateien gespeichert: {x_filepath}, {y_filepath}")
        print(f"Label-Mapping gespeichert: {mapping_filepath}")
