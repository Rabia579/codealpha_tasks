# CodeAlpha_HandwrittenCharacterRecognition

**CodeAlpha Machine Learning Internship — Task: Handwritten Character Recognition**

A Convolutional Neural Network (CNN) that classifies handwritten digits and letters using the EMNIST Balanced dataset (47 classes).

---

## 🎯 Objective

Identify handwritten characters or alphabets using image processing and deep learning, as specified in the CodeAlpha ML internship task list.

## 📊 Dataset

**EMNIST Balanced** — 131,600 handwritten character images (28×28 grayscale), 47 classes covering:
- Digits `0–9`
- Uppercase letters `A–Z`
- A subset of lowercase letters whose shape is visually distinct from their uppercase form (`a, b, d, e, f, g, h, n, q, r, t`)

Visually near-identical case pairs (e.g. `C`/`c`, `O`/`o`, `S`/`s`) are merged into a single class in this split, which is what makes it "balanced" and harder than plain MNIST.

- Train: 112,800 images (further split 90/10 into train/validation, stratified)
- Test: 18,800 images (held out completely — used exactly once, for final evaluation only)

**Source:** downloaded directly from a [Hugging Face mirror](https://huggingface.co/datasets/Royc30ne/emnist-balanced) of the official [NIST EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), in the original gzip'd IDX binary format. This sidesteps two reliability issues encountered with more common loading methods in Google Colab: `tensorflow_datasets` pulling in a `protobuf`/`tensorflow-metadata` version conflict, and the `emnist` pip package's default NIST download host being intermittently unavailable.

## 🧠 Model Architecture

Two CNNs with identical architecture, trained and compared (see Methodology below):

```
Input (28x28x1)
[Augmented variant only: RandomRotation / RandomTranslation / RandomZoom]
→ Conv2D(32) → Conv2D(32) → MaxPool2D → Dropout(0.25)
→ Conv2D(64) → Conv2D(64) → MaxPool2D → Dropout(0.25)
→ Flatten
→ Dense(256) → BatchNorm → Dropout(0.5)
→ Dense(47, softmax)
```

~881K parameters. Optimizer: `Adam(learning_rate=1e-3)`, explicit rather than the default `'adam'` string. Loss: sparse categorical cross-entropy. Trained with `EarlyStopping` and `ReduceLROnPlateau`, both monitoring a held-out **validation** split (10% of training data, stratified) — never the test set.

## 📈 Results

| Metric | Score |
|---|---|
| **Final model selected** | Baseline (no augmentation) — selected on validation accuracy |
| Test Accuracy | **90.04%** |
| Test Loss | 0.2891 |
| Accuracy (overall) | 0.90 |
| Macro avg F1 | ~0.90 |
| Weighted avg F1 | ~0.90 |

*(All 47 classes have equal support — 400 test samples each — so macro and weighted averages are effectively identical.)*

### Methodology note: how the final model was chosen

Two CNNs are trained — a baseline and one with data augmentation (random rotation/zoom/shift). They're compared using **validation accuracy only** (Section 7); the test set is never used to choose between them. The winning model is then evaluated on the held-out test set **exactly once** (Section 8) — that single run produces the official test accuracy, confusion matrix, classification report, and per-class accuracy reported above and in the notebook. This avoids test-set leakage into model-selection decisions, which would otherwise make the reported accuracy optimistic.

**Result of the experiment:** the baseline model outperformed the augmented one on validation accuracy throughout training (baseline: ~90–92% val acc vs. augmented: ~88% val acc, with the augmented model also showing consistently higher validation loss). Since EMNIST characters are already centered and fairly consistent in scale, the rotation/zoom/shift augmentation likely made the training task harder without a compensating robustness benefit on this particular test distribution — a legitimate, reportable outcome rather than a failed experiment. The baseline was selected accordingly.

### Where the errors concentrate

Nearly all misclassifications fall on a small set of characters that are genuinely ambiguous in handwriting — a known, well-documented property of EMNIST Balanced rather than a model weakness. In this run, the weakest classes were:

| Character | Per-class accuracy | F1-score | Commonly confused with |
|---|---|---|---|
| L | 0.52 | 0.58 | I, 1 |
| F | 0.62 | 0.67 | f |
| O | 0.63 | 0.68 | 0 |
| q | 0.64 | — | g, 9 |
| I | 0.66 | 0.69 | L, 1 |
| g | 0.69 | — | q, 9 |
| f | 0.72 | — | F |
| 1 | 0.74 | 0.65 | I, L |
| 0 | 0.77 | 0.71 | O |
| 9 | 0.83 | 0.76 | g, q |

By contrast, visually distinctive characters (`R`, `W`, `H`, `A`, `7`, `3`, `M`, `N`, `K`, `E`, `B`) scored 0.97–1.00.

### Real-world handwriting testing

Section 10 lets you upload your own photo of a handwritten character and see the model's top-3 predictions. Preprocessing includes automatic bounding-box cropping and contrast stretching (`ImageOps.autocontrast`), since phone photos are often lower-contrast than EMNIST's clean training images — mid-gray ink from shadows or uneven lighting otherwise reads as "uncertain" to the model.

Section 11 also includes a live Gradio drawing demo. Worth noting: Sketchpad-drawn characters (thin, uniform-width mouse strokes) look visually different from EMNIST's pen/pencil-derived training images, so predictions on drawn input are occasionally less reliable than on photographed handwriting — an honest, explainable limitation rather than a bug.

## 🚀 Running the Notebook

1. Open `CodeAlpha_HandwrittenCharacterRecognition.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set `Runtime > Change runtime type > GPU`
3. Run all cells top to bottom:
   - Sections 1–3: install, download EMNIST, preprocess, split train/val/test
   - Sections 4–5: train the baseline and augmented CNNs
   - Sections 6–8: compare on validation, select the winner, evaluate it on the test set once
   - Section 9: save the model
   - Section 10: (optional) test on your own handwriting photos
   - Section 11: deployment — saves `inference.py`, `class_names.json`, and the final `.keras` model; optionally launches a live Gradio demo
4. The trained model saves as `handwritten_character_recognition_deploy.keras`

### Running the Gradio demo

After running Section 11's Gradio cell, Colab prints a public `https://xxxxx.gradio.live` link (valid for 72 hours) — open it, draw a character on the sketchpad, and see live predictions. This is the fastest way to demo the model without recording your screen inside the notebook.

### Screenshots
Validation accuracy/loss comparison used to select the final model 
<img width="1487" height="487" alt="image" src="https://github.com/user-attachments/assets/d871e680-1ef1-4cc2-9342-db441b6bc169" />

EMNIST Balanced test set confusion matrix 
<img width="1312" height="775" alt="confu" src="https://github.com/user-attachments/assets/6464b931-448e-4e2b-b51c-b38a91bf4246" />

Per-class accuracy, weakest classes highlighted in red 

<img width="1486" height="740" alt="per" src="https://github.com/user-attachments/assets/56a6f064-c944-4dd9-a8ea-58f7710168c1" />

Live handwriting recognition demo 

<img width="1130" height="611" alt="b" src="https://github.com/user-attachments/assets/10397550-0e50-446b-a78c-3bec284b4530" />


## 🔭 Extending to Words / Sentences

This project handles single-character classification. Extending to full word/sentence recognition would require a different pipeline — notes and a recommended approach (CRNN + CTC loss, trained on the IAM Handwriting Database) are included in Section 12 of the notebook.

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn (train/val split, evaluation metrics)
- Gradio (optional live demo)

## 📄 About CodeAlpha

This project was completed as part of the [CodeAlpha](https://www.codealpha.tech) Machine Learning Internship program.

---
