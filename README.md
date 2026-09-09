# CNN for MNIST Digit Classification

A Convolutional Neural Network (CNN) that classifies handwritten digits (0–9) from the MNIST dataset (70,000 grayscale 28×28 images: 60,000 training / 10,000 test). This project demonstrates a complete deep learning pipeline — data preprocessing, model design, training, evaluation, and inference — using core CNN building blocks such as convolution, pooling, dropout, and batch normalization.

---

## 🧰 Framework

> Delete whichever section doesn't apply.

**Option A — TensorFlow / Keras**
Built using `tensorflow.keras`, leveraging the Sequential/Functional API for rapid model definition, `ImageDataGenerator` (or `tf.data`) for preprocessing/augmentation, and Keras callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`) for training control.

**Option B — PyTorch**
Built using `torch.nn.Module` for custom model definition, `torch.utils.data.DataLoader` for batching, and a manual training loop with `torch.optim` (Adam/SGD) and `torch.nn.CrossEntropyLoss`.

```
Framework: TensorFlow 2.x (Keras)   |   PyTorch 2.x
Language:  Python 3.10+
```

---

## 🏗️ Architecture Details

A typical, well-performing CNN for MNIST looks like this — adjust to match your actual model:

| Layer | Type | Details |
|---|---|---|
| 1 | Conv2D | 32 filters, 3×3 kernel, ReLU, padding='same' |
| 2 | Conv2D | 32 filters, 3×3 kernel, ReLU |
| 3 | MaxPooling2D | 2×2 pool size |
| 4 | Dropout | rate = 0.25 |
| 5 | Conv2D | 64 filters, 3×3 kernel, ReLU, padding='same' |
| 6 | Conv2D | 64 filters, 3×3 kernel, ReLU |
| 7 | MaxPooling2D | 2×2 pool size |
| 8 | Dropout | rate = 0.25 |
| 9 | Flatten | — |
| 10 | Dense | 256 units, ReLU |
| 11 | Dropout | rate = 0.5 |
| 12 | Dense (output) | 10 units, Softmax |

**Design notes:**
- Two conv blocks (32 → 64 filters) allow the network to learn low-level edges first, then higher-level digit shapes.
- 3×3 kernels are used throughout for a good accuracy/efficiency tradeoff.
- MaxPooling after each block reduces spatial dimensions and controls overfitting.
- Dropout layers (0.25 after conv blocks, 0.5 before output) reduce overfitting on the relatively small dataset.
- Batch Normalization can optionally be added after each Conv2D layer to stabilize and speed up training.
- Total parameters: **~[XXX,XXX]** *(fill in from `model.summary()` / `torchsummary`)*

**Training configuration:**
```
Optimizer:      Adam (lr=0.001)
Loss function:  Categorical Crossentropy / CrossEntropyLoss
Batch size:     128
Epochs:         [XX]
LR schedule:    ReduceLROnPlateau (optional)
Data augment:   Random rotation (±10°), zoom (0.1), width/height shift (0.1)
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | **XX.XX%** |
| Validation Accuracy | **XX.XX%** |
| **Test Accuracy** | **XX.XX%** *(fill in after training)* |
| Test Loss | X.XXXX |
| Parameters | XXX,XXX |
| Training Time | XX min (on [GPU/CPU model]) |

*(Optional: add a confusion matrix image, accuracy/loss curves, or sample predictions here.)*

---

## 📁 Project Structure

```
├── data/               # MNIST dataset (auto-downloaded or cached)
├── models/             # Saved model checkpoints (.h5 / .pt)
├── notebooks/          # Jupyter notebooks for EDA & experimentation
├── src/
│   ├── model.py        # CNN architecture definition
│   ├── train.py        # Training loop / script
│   ├── evaluate.py     # Evaluation on test set
│   └── predict.py       # Inference on custom images
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/cnn-mnist.git
cd cnn-mnist

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py --epochs 20 --batch-size 128

# Evaluate on the test set
python src/evaluate.py --model models/best_model.h5
```

---

## 🔮 Future Improvements
- Experiment with deeper architectures (ResNet-style blocks) for marginal accuracy gains
- Add TensorBoard / Weights & Biases logging
- Deploy as a simple web app (Flask/Streamlit) for live digit-drawing inference
- Export to ONNX/TFLite for mobile deployment

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
