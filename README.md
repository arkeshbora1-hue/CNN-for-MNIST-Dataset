✍️ Handwritten Digit Recognition — MNIST

A neural network that reads handwriting the way a human eye scans a page.

Show Image Show Image Show Image Show Image

</div>
🧠 What this is

Every digit you scribble on paper — a 7, a 3, a 9 — has a shape your brain recognizes instantly. This project teaches a neural network to do the same thing, using the classic MNIST dataset: 70,000 grayscale images of handwritten digits, collected from real human handwriting.

The model looks at a 28×28 pixel image and answers one question: "Which digit is this?" — and gets it right 97.33% of the time on data it's never seen before.

⚡ Quick facts
	
🧩 Task	Multi-class image classification (10 classes: digits 0–9)
🗂️ Dataset	MNIST — 60,000 train / 10,000 test images, 28×28 grayscale
🏗️ Model type	Fully-connected (dense) neural network
🎯 Test accuracy	97.33%
⏱️ Training time	30 epochs, batch size 128
🛠️ Built with	TensorFlow / Keras
🔍 How it works
Raw image (28×28)
        │
        ▼
  Flatten → 784 values
        │
        ▼
  Standardize (mean=0, std=1)
        │
        ▼
  Dense layer · 128 neurons · ReLU
        │
        ▼
  Dense layer · 64 neurons · ReLU
        │
        ▼
  Dense layer · 10 neurons · Softmax
        │
        ▼
  Prediction: "This is a 7" (or whichever digit scores highest)

Under the hood, this is a classic Artificial Neural Network (ANN) — no convolutional layers yet, just fully-connected layers learning patterns directly from pixel intensities. It's a great baseline before stepping up to a true CNN.

💡 Curious about CNNs? A convolutional version of this same problem (using Conv2D + MaxPooling2D layers) typically pushes accuracy above 99% by learning spatial patterns like edges and curves instead of raw pixel values. See What's next below.

📈 How well does it actually work?
Reaches ~99% training accuracy and 97.33% test accuracy
Misclassifies fewer than 3 in every 100 unseen digits
Struggles the most distinguishing visually similar digits (e.g. 4 vs 9, 3 vs 5) — visible in the confusion matrix generated in the notebook
Precision, recall, and F1-score all sit in the 0.96–0.99 range across every digit class
🚀 Try it yourself
bash
git clone https://github.com/<your-username>/mnist-digit-recognition.git
cd mnist-digit-recognition
pip install tensorflow scikit-learn matplotlib seaborn numpy
jupyter notebook CNN_for_MNIST_Dataset_CONVOLUTIONAL_NEURAL_NETWORK_FOR_DEEP_LEARNING_BASICS_.ipynb

Load the pre-trained model instead of retraining:

python
from tensorflow.keras.models import load_model
model = load_model("mnist_ann.keras")
prediction = model.predict(your_image)
🗺️ What's next
 Swap in Conv2D + MaxPooling2D layers for a true CNN architecture
 Add Dropout / Batch Normalization to close the train–test accuracy gap
 Augment training data (rotation, shift, zoom) for better generalization
 Wrap the model in a small Streamlit app for live digit-drawing predictions
 Export to TFLite/ONNX for mobile or edge deployment
📄 License

MIT — free to use, modify, and learn from.

<div align="center">

Built as a hands-on deep learning fundamentals exercise.

</div>
