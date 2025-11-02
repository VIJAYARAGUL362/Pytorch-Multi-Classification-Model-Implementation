# PyTorch Multiclass Classification

A neural network implementation for multiclass classification using PyTorch, demonstrating fundamental deep learning concepts including model architecture design, training loops, and decision boundary visualization.

## 🎯 Project Overview

This project implements a feedforward neural network to classify data points into multiple classes using PyTorch. The model is trained on synthetic blob data and achieves high accuracy through proper architecture design and optimization techniques.

## 🚀 Key Features

- **Custom Neural Network Architecture**: Sequential model with configurable hidden layers
- **GPU Acceleration**: Device-agnostic code for automatic GPU utilization when available
- **Comprehensive Training Pipeline**: Full implementation of training and evaluation loops
- **Visual Analytics**: Decision boundary visualization to understand model behavior
- **Performance Metrics**: Integration with scikit-learn for accuracy evaluation

## 🏗️ Model Architecture

```
Input Layer (2 features)
    ↓
Hidden Layer 1 (128 units) + ReLU
    ↓
Hidden Layer 2 (128 units) + ReLU
    ↓
Output Layer (4 classes)
```

The model uses:
- **Activation Function**: ReLU for hidden layers
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (learning rate: 0.01)

## 📊 Dataset

- **Type**: Synthetic blob data generated using scikit-learn
- **Features**: 2D features for easy visualization
- **Classes**: 4 distinct clusters
- **Samples**: 1000 data points
- **Split**: 80% training, 20% testing

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **scikit-learn**: Dataset generation and metrics
- **torchsummary**: Model architecture summary

## 📋 Requirements

```
torch
numpy
matplotlib
scikit-learn
torchsummary
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd pytorch-multiclass-classification
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib scikit-learn torchsummary
```

## 💻 Usage

Run the Jupyter notebook or Python script:

```bash
python pytorch_multiclass_classification.py
```

The script will:
1. Generate synthetic classification data
2. Split into training and testing sets
3. Initialize and train the neural network
4. Display training progress with loss and accuracy metrics
5. Visualize decision boundaries for both train and test sets

## 📈 Model Training

The training loop implements:
- **Forward propagation**: Computing predictions
- **Loss calculation**: Using CrossEntropyLoss
- **Backpropagation**: Calculating gradients
- **Parameter updates**: Using Adam optimizer
- **Evaluation**: Testing on held-out data

Training progress is displayed every epoch with:
- Training loss
- Testing loss
- Training accuracy
- Testing accuracy

## 🎨 Visualization

The project includes decision boundary plots that show:
- How the model separates different classes
- Model confidence across the feature space
- Training vs. testing data distribution
- Classification regions for each class

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- PyTorch fundamentals and tensor operations
- Neural network architecture design
- Training loop implementation
- GPU acceleration techniques
- Model evaluation and metrics
- Data visualization for ML models

## 🙏 Acknowledgments

This project was developed following Daniel Bourke's PyTorch Deep Learning course. The helper functions and learning approach are adapted from his excellent tutorial series.

## 📝 License

MIT License - feel free to use this code for learning and development purposes.

## 📧 Contact

Feel free to reach out if you have questions or suggestions for improvements!

---

**Note**: This is a learning project demonstrating core PyTorch concepts. For production use, consider additional features like model checkpointing, hyperparameter tuning, and cross-validation.
