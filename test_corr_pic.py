import os
from PIL import Image

def check_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                with Image.open(os.path.join(root, file)) as img:
                    img.verify()  # Überprüfe, ob das Bild gültig ist
            except Exception as e:
                print(f"Fehler beim Laden des Bildes {file}: {e}")

check_images("cats_and_dogs_small/train")