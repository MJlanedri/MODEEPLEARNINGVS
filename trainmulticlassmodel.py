from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import numpy as np
from BinaryDatasetCreator import BinaryDatasetCreator


def train_multiclass_model(
    json_file,
    image_dir,
    output_dataset_dir,
    output_model_dir,
    outputs_mapping_labeles,
    input_shape,
    num_classes,
    build_model_fn,
    optimizer,
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
    activation_cls="relu",
    dropout_rate=0.5,
    use_batch_normalization=True,
    use_dense=True,
    use_global_pooling=False,
    batch_size=32,
    epochs=10,
):
    """
    Trainiert ein Modell mit mehreren Klassen basierend auf den gelabelten Bildern.

    Args:
        json_file (str): Pfad zur JSON-Datei mit Labels.
        image_dir (str): Verzeichnis mit Bildern.
        output_dir (str): Verzeichnis, um die Binärdateien zu speichern.
        input_shape (tuple): Eingabeform des Modells (height, width, channels).
        num_classes (int): Anzahl der Klassen.
        build_model_fn (callable): Funktion, die das Modell erstellt.
        optimizer (callable): Optimierer (z.B. Adam).
        Weitere Parameter: Konfiguration für das Modell.
    """
    # Erstelle den Dataset-Creator und speichere die Daten
    creator = BinaryDatasetCreator(json_file=json_file, image_dir=image_dir, img_shape=input_shape)
    creator.save_binary_files(outputs_mapping_labeles)

    # Lade die verarbeiteten Daten
    x = np.load(os.path.join(outputs_mapping_labeles, "x.npy"))
    y = np.load(os.path.join(outputs_mapping_labeles, "y.npy"))

    # One-Hot-Codierung der Labels
    y_categorical = to_categorical(y, num_classes=num_classes)

    # Teile die Daten in Training, Validierung und Test
    X_train, X_temp, y_train, y_temp = train_test_split(x, y_categorical, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Erstelle das Modell mit den angegebenen Parametern
    model = build_model_fn(
        optimizer=optimizer,
        learning_rate=learning_rate,
        filter_block1=filter_block1,
        kernel_size_block1=kernel_size_block1,
        filter_block2=filter_block2,
        kernel_size_block2=kernel_size_block2,
        filter_block3=filter_block3,
        kernel_size_block3=kernel_size_block3,
        filter_block4=filter_block4,
        kernel_size_block4=kernel_size_block4,
        dense_layer_size=dense_layer_size,
        kernel_initializer=kernel_initializer,
        activation_cls=activation_cls,
        dropout_rate=dropout_rate,
        use_batch_normalization=use_batch_normalization,
        use_dense=use_dense,
        use_global_pooling=use_global_pooling,
        num_classes=num_classes,
    )

    # Trainiere das Modell
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
    )

    # Evaluierung auf den Testdaten
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Testgenauigkeit: {accuracy * 100:.2f}%")

    # Speichere das Modell
    model_path = os.path.join(output_model_dir, "image_classifier.h5")
    model.save(model_path)
    print(f"Modell gespeichert unter: {model_path}")

    return history, model
