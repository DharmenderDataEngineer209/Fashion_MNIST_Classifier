# Fashion MNIST Classifier

This project is a **Fashion MNIST Classifier** that predicts clothing categories using a deep learning model built with TensorFlow and Keras. It’s designed to help understand image classification with neural networks in a simple, practical way.

---

## Features

Here’s what this project does:

- **Loads and explores data**: The Fashion MNIST dataset contains 70,000 grayscale images of 10 clothing categories (e.g., shirts, sneakers).
- **Builds a neural network**: A straightforward architecture with three dense layers, each serving a specific purpose.
- **Trains and evaluates the model**: Tracks accuracy and loss over 30 epochs with validation.
- **Makes predictions**: Tests the model on unseen data and identifies the clothing items.

---

## Project Workflow

### 1. Data Preparation

- Split data into training, validation, and test sets.
- Normalize pixel values to fall between 0 and 1 for faster training.

### 2. Model Architecture

The neural network consists of:

- **Flatten Layer**: Converts 28x28 images into 1D arrays.
- **Two Dense Layers**: Each with ReLU activation for learning complex patterns.
- **Output Layer**: A softmax layer with 10 units to classify the images into 10 categories.

### 3. Model Training

- **Loss Function**: Sparse categorical cross-entropy to handle multi-class labels.
- **Optimizer**: Stochastic Gradient Descent (SGD) for efficient weight updates.
- **Metrics**: Accuracy to measure performance.

### 4. Predictions

- Predicts clothing labels on test data.
- Uses `np.argmax` to convert probabilities into class labels.

---

## Why This Project?

Deep learning can seem complex. This project simplifies it. By focusing on a real-world dataset like Fashion MNIST, you’ll see how neural networks recognize patterns in images.

---

## Tools and Skills Used

- **TensorFlow/Keras**: For building and training the model.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualizing images and results.

---

## How to Run

1. Clone the repository.
2. Install required libraries: `tensorflow`, `numpy`, `matplotlib`.
3. Run the script to train the model and make predictions.

---

## Explore the Code

Check out how the neural network processes data, learns from it, and predicts outcomes. Play around with the architecture or hyperparameters to improve results.

Happy coding!
