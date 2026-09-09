# MNIST Digit Classification — Neural Network

A feed-forward (fully-connected) neural network built with **TensorFlow / Keras** that classifies handwritten digits (0–9) from the MNIST dataset (70,000 grayscale 28×28 images: 60,000 train / 10,000 test). The pipeline covers data preprocessing, model training, evaluation, and visualization of results.

> **Note:** This notebook implements a Dense (ANN) architecture — each image is flattened to a 784-length vector before being passed through fully-connected layers. It does not currently use Conv2D/MaxPooling layers. If you're looking to extend this into a true CNN, see the *Future Improvements* section below.

---

## 🧰 Framework

```
Framework: TensorFlow 2.x (Keras Sequential API)
Language:  Python 3.13
Libraries: NumPy, scikit-learn, Matplotlib, Seaborn
```

---

## 🔄 Data Pipeline

1. **Load** — MNIST loaded via `tf.keras.datasets.mnist.load_data()`
2. **Flatten** — Images reshaped from `(28, 28)` → `(784,)`
3. **Split** — 90% train / 10% validation, stratified by digit class (`train_test_split`, `random_state=42`)
   - Train: 54,000 images
   - Validation: 6,000 images
   - Test: 10,000 images
4. **Scale** — Features standardized with `StandardScaler` (fit on train, applied to val/test)
5. **Encode labels** — One-hot encoded via `to_categorical` (10 classes)

---

## 🏗️ Architecture

| Layer | Type | Output Shape | Params |
|---|---|---|---|
| 1 | Dense | (None, 128) | 100,480 |
| 2 | Dense | (None, 64) | 8,256 |
| 3 | Dense (output) | (None, 10) | 650 |

```
Input:        784 (flattened 28×28 image)
Hidden 1:     Dense(128, activation='relu')
Hidden 2:     Dense(64, activation='relu')
Output:       Dense(10, activation='softmax')

Total params:     109,386 (427.29 KB)
Trainable params: 109,386
```

**Training configuration:**
```
Optimizer:      Adam
Loss function:  Categorical Crossentropy
Metric:         Accuracy
Batch size:     128
Epochs:         30
Validation:     6,000 held-out images
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy (final epoch) | ~99% |
| Validation Accuracy | ~96–97% |
| **Test Accuracy** | **97.33%** |
| Test Loss | 0.2616 |
| Parameters | 109,386 |

**Per-class performance** (from `classification_report`): precision, recall, and f1-score are all in the **0.96–0.99** range across all 10 digit classes, with digits 0, 1, and 6 performing strongest and 5 slightly weaker relative to the rest.

Additional evaluation artifacts generated in the notebook:
- Training/validation accuracy and loss curves over 30 epochs
- Confusion matrix (Seaborn heatmap) across all 10 digit classes
- Sample prediction grids comparing predicted vs. actual labels

---

## 📁 Project Structure

```
├── CNN_for_MNIST_Dataset_CONVOLUTIONAL_NEURAL_NETWORK_FOR_DEEP_LEARNING_BASICS_.ipynb
├── mnist_ann.keras     # Saved trained model
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/mnist-neural-network.git
cd mnist-neural-network

# Install dependencies
pip install tensorflow scikit-learn matplotlib seaborn numpy

# Run the notebook
jupyter notebook CNN_for_MNIST_Dataset_CONVOLUTIONAL_NEURAL_NETWORK_FOR_DEEP_LEARNING_BASICS_.ipynb
```

The trained model is saved to `mnist_ann.keras` and can be reloaded with:
```python
from tensorflow.keras.models import load_model
model = load_model("mnist_ann.keras")
```

---

## 🔮 Future Improvements
- **Add real convolutional layers** (Conv2D + MaxPooling2D blocks) to build an actual CNN, which typically pushes MNIST accuracy above 99%
- Add Dropout / Batch Normalization to reduce the train/test accuracy gap (currently training accuracy noticeably exceeds test accuracy, suggesting mild overfitting)
- Add data augmentation (rotation, shift, zoom) to improve generalization
- Log experiments with TensorBoard or Weights & Biases
- Deploy as a simple web app (Flask/Streamlit) for live digit-drawing inference

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
