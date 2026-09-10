"""
Standalone inference module for the CodeAlpha handwritten character
recognition model. Usage:

    from inference import HandwritingClassifier
    clf = HandwritingClassifier('handwritten_character_recognition_deploy.keras',
                                 'class_names.json')
    predictions = clf.predict('path/to/photo.png')
"""
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class HandwritingClassifier:
    def __init__(self, model_path: str, class_names_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(class_names_path) as f:
            self.class_names = json.load(f)

    @staticmethod
    def _preprocess(img: Image.Image) -> np.ndarray:
        img = img.convert('L')

        # Stretch contrast first: phone-photo ink is often mid-gray rather
        # than the near-pure black/white EMNIST was trained on.
        img = ImageOps.autocontrast(img, cutoff=2)

        if np.array(img).mean() > 127:
            img = ImageOps.invert(img)
        arr = np.array(img)

        mask = arr > 30
        if mask.any():
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            arr = arr[r0:r1 + 1, c0:c1 + 1]

        h, w = arr.shape
        size = max(h, w)
        pad_h, pad_w = (size - h) // 2, (size - w) // 2
        arr = np.pad(arr, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)))
        margin = size // 5
        arr = np.pad(arr, margin)

        img = Image.fromarray(arr).resize((28, 28))
        img = ImageOps.autocontrast(img, cutoff=1)  # re-stretch the final crop
        arr = np.array(img).astype('float32') / 255.0
        return arr.reshape(1, 28, 28, 1)

    def predict(self, image_path: str, top_k: int = 3):
        img = Image.open(image_path)
        arr = self._preprocess(img)
        preds = self.model.predict(arr, verbose=0)[0]
        top_idx = preds.argsort()[-top_k:][::-1]
        return [
            {"character": self.class_names[i], "confidence": float(preds[i])}
            for i in top_idx
        ]


if __name__ == "__main__":
    import sys
    clf = HandwritingClassifier(
        "handwritten_character_recognition_deploy.keras", "class_names.json"
    )
    result = clf.predict(sys.argv[1])
    print(result)
