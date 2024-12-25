# Project Title: Enhanced CNN Models for Image Classification

## Overview
This project explores three Convolutional Neural Network (CNN) models for image classification:
1. **ResNet-50** - A deep residual network designed to overcome the vanishing gradient problem.
2. **LeNet-5** - A classic CNN known for its simplicity and efficiency in basic image classification tasks.
3. **CNN_20** - A proposed custom model with enhanced performance through deeper architecture and regularization techniques.

## Models
### Model 1: ResNet-50
- **Purpose**: Handles large image datasets with its "residual learning" framework.
- **Architecture**: 50 layers with convolutional, batch normalization, and residual blocks.
- **Key Features**:
  - Residual blocks to ease training of deeper networks.
  - Fully connected layers for classification.
  - ReLU activation for non-linearity.

### Model 2: LeNet-5
- **Purpose**: Ideal for simple datasets, originally designed for digit recognition.
- **Architecture**: Combination of convolutional and pooling layers leading to fully connected layers.
- **Key Features**:
  - Pooling for dimensionality reduction.
  - Simple activation functions (Sigmoid or Tanh).
  - Efficient for limited computational resources.

### Model 3: CNN_20 (Proposed)
- **Purpose**: Custom model with 20 convolutional layers, dropout for regularization, and batch normalization for improved training stability.
- **Key Features**:
  - Deep architecture to capture granular details.
  - Dropout (p = 0.5) to prevent overfitting.
  - Max pooling for dimensionality reduction.

## Results
- **ResNet-50**: Accuracy of 90% on CIFAR-10 dataset.
- **LeNet-5**: Accuracy of 66%, suitable for simpler datasets.
- **CNN_20**: Accuracy of 79%, outperforming LeNet-5 and demonstrating strong potential for complex datasets.

## Performance Metrics
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Sensitivity in capturing relevant cases.
- **F1 Score**: Balance between precision and recall.

## Future Work
- **Optimization**: Hyperparameter tuning for CNN_20.
- **Data Augmentation**: Enhance datasets with augmented samples.
- **Deployment**: Real-world testing on portable devices like Raspberry Pi.

## Setup Instructions
1. Clone the repository.
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or scripts for training and evaluation.

## Hardware and Software Requirements
- **Hardware**: NVIDIA GPU for training, Raspberry Pi for deployment.
- **Software**: Python, TensorFlow, PyTorch, OpenCV.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

