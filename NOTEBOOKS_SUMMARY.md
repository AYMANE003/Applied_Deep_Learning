# Notebooks Content Summary

## Overview
This document provides a detailed summary of each notebook's content, objectives, and key learnings.

---

## 1. Linear Regression: From Scratch to TensorFlow
**File**: `Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb`

### Objectives
- Understand linear regression from first principles
- Implement gradient descent manually
- Compare three different approaches
- Learn model evaluation metrics

### Topics Covered
1. **Mathematical Foundations**
   - Linear equation: y = mx + b
   - Cost function (Mean Squared Error)
   - Gradient descent algorithm
   - Learning rate and convergence

2. **Implementation Approaches**
   - From-scratch implementation with NumPy
   - Scikit-Learn Linear Regression
   - TensorFlow/Keras implementation
   
3. **Model Evaluation**
   - R² Score
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)

4. **Data Handling**
   - Train-test split
   - Feature normalization
   - Handling missing data

### Key Concepts
- Supervised learning basics
- Hyperparameter tuning
- Model comparison and selection
- Visualization of predictions

### Libraries Used
`numpy`, `pandas`, `sklearn`, `tensorflow`, `matplotlib`, `seaborn`

### Output/Results
- Comparison plots of three models
- Prediction accuracy metrics
- Error visualizations

---

## 2. Artificial Neural Networks: ReLU vs Sigmoid
**File**: `Livrable_2_RNA_(RL+ReLu_Sigmoid).ipynb`

### Objectives
- Understand artificial neural network architecture
- Compare ReLU and Sigmoid activation functions
- Visualize decision boundaries
- Analyze performance on different datasets

### Topics Covered
1. **Neural Network Basics**
   - Perceptron model
   - Multilayer networks
   - Forward propagation
   - Backpropagation

2. **Activation Functions**
   - Sigmoid activation
   - ReLU (Rectified Linear Unit)
   - Comparisons and use cases
   - Gradient flow analysis

3. **Classification Tasks**
   - Binary classification
   - Decision boundaries visualization
   - Non-linear separability

4. **Test Datasets**
   - Gaussian blobs (well-separated)
   - Less separable clusters
   - Moon dataset (crescent shapes)
   - Helix dataset (spiral pattern)

### Key Concepts
- Neural network training
- Activation function impact on performance
- Handling non-linearly separable data
- Training dynamics visualization

### Libraries Used
`numpy`, `sklearn`, `matplotlib`, `seaborn`

### Output/Results
- Decision boundary plots for each activation
- Performance metrics comparison
- Training convergence curves
- 3D visualization of neural network decisions

---

## 3. Convolutional Neural Networks: Fundamentals
**File**: `Livrable_3_CNN_solo.ipynb`

### Objectives
- Understand convolution operation basics
- Learn filter/kernel concepts
- Explore stride and padding effects
- Visualize feature maps

### Topics Covered
1. **Convolution Basics**
   - Convolution operation mechanics
   - Filter/kernel concept
   - Feature extraction
   - Element-wise multiplication and summation

2. **CNN Parameters**
   - Stride (step size)
   - Padding (same/valid)
   - Number of filters
   - Output dimension calculation

3. **Pooling Operations**
   - Max pooling
   - Average pooling
   - Dimensionality reduction

4. **Feature Visualization**
   - Feature maps from filters
   - Activation maps
   - Spatial hierarchy

### Configuration Used
- Input: 5×5×3 RGB image
- Filters: 2 filters of 3×3×3
- Stride: 2
- Padding: 1

### Key Concepts
- Image processing with convolutions
- Feature extraction without hand-crafting
- Spatial hierarchies in images
- Parameter sharing in CNNs

### Libraries Used
`tensorflow`, `numpy`, `matplotlib`, `PIL`

### Output/Results
- Feature maps visualization
- Filter activation patterns
- Effect of different parameters
- Dimension calculations

---

## 4. CNN Application: MNIST Digit Classification
**File**: `Livrable_4_CNN_Mnist_Dataset.ipynb`

### Objectives
- Build and train a practical CNN
- Classify handwritten digits
- Evaluate classification performance
- Analyze model predictions

### Topics Covered
1. **Dataset Handling**
   - MNIST dataset (70,000 samples)
   - Data normalization
   - Train-validation-test split
   - Class distribution

2. **CNN Architecture**
   - Multiple convolutional layers
   - Pooling layers
   - Fully connected layers
   - Dropout for regularization

3. **Training Process**
   - Loss function (categorical cross-entropy)
   - Optimizer (Adam)
   - Learning rate scheduling
   - Early stopping

4. **Evaluation**
   - Accuracy metrics
   - Confusion matrices
   - Per-class performance
   - Misclassification analysis

5. **Model Analysis**
   - Correct vs incorrect predictions
   - Confidence scores
   - Failure case analysis

### Key Concepts
- End-to-end deep learning pipeline
- Data preprocessing for images
- Model training and validation
- Performance metrics interpretation
- Overfitting detection

### Libraries Used
`tensorflow`, `keras`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`

### Output/Results
- Model accuracy (>95%)
- Confusion matrix heatmap
- Sample predictions with confidence
- Training/validation curves
- Misclassification examples

---

## 5. Natural Language Processing: Sarcasm Detection
**File**: `Livrable_6_Sarcasm_NLP.ipynb`

### Objectives
- Understand NLP fundamentals
- Build text classification model
- Handle sequential data
- Detect sarcasm in headlines

### Topics Covered
1. **Text Preprocessing**
   - Tokenization
   - Lowercasing
   - Removing special characters
   - Padding sequences

2. **Text Representation**
   - Word embeddings
   - Embedding layers
   - Token indexing
   - Vocabulary creation

3. **Model Architectures**
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - Bidirectional layers
   - Attention mechanisms

4. **Classification Approach**
   - Binary classification (sarcasm/not sarcasm)
   - Class imbalance handling
   - Probability interpretation

5. **Text Analysis**
   - Key word importance
   - Feature extraction from text
   - Sentiment vs sarcasm

### Key Concepts
- Sequential data processing
- Recurrent neural networks
- Text embeddings
- Sequence-to-sequence learning
- Natural language understanding

### Libraries Used
`tensorflow`, `keras`, `nltk`, `numpy`, `pandas`, `matplotlib`

### Output/Results
- Classification accuracy metrics
- ROC curves
- Confusion matrix
- Sample predictions
- Word importance analysis
- Model interpretability

---

## 6. Time Series Forecasting (Folder: Livrable_5_Time_Series)

### Overview
A comprehensive study of time series forecasting with three interconnected notebooks progressing from basics to advanced methods.

---

### 6.1 Time Series Fundamentals
**File**: `Time_Series_Séance1.ipynb`

#### Objectives
- Understand time series components
- Test for stationarity
- Implement basic forecasting methods
- Visualize temporal patterns

#### Topics Covered
1. **Time Series Components**
   - Trend (long-term direction)
   - Seasonality (repeating patterns)
   - Cyclicity (longer repeating cycles)
   - Residuals (noise)

2. **Stationarity**
   - Augmented Dickey-Fuller (ADF) test
   - Differencing
   - Detrending
   - Seasonal decomposition

3. **Basic Forecasting Methods**
   - Moving average
   - Exponential smoothing
   - ARIMA components
   - Naive methods

4. **Visualization Techniques**
   - Time series plots
   - Decomposition plots
   - ACF/PACF plots
   - Lag plots

#### Key Concepts
- Temporal dependencies
- Autocorrelation analysis
- Trend and seasonal extraction
- Stationarity requirement for modeling

#### Libraries Used
`pandas`, `numpy`, `matplotlib`, `statsmodels`, `sklearn`

#### Output/Results
- Decomposition visualizations
- Stationarity test results
- Forecast plots
- Error metrics (MAE, RMSE, MAPE)

---

### 6.2 Time Series Model Comparison
**File**: `Time_Series_Séance2_models_comparision.ipynb`

#### Objectives
- Compare multiple forecasting models
- Evaluate performance metrics
- Implement cross-validation for series
- Analyze model strengths/weaknesses

#### Topics Covered
1. **Classical Methods**
   - ARIMA (AutoRegressive Integrated Moving Average)
   - SARIMA (Seasonal ARIMA)
   - Exponential Smoothing variants
   - Prophet (Facebook's method)

2. **Model Evaluation**
   - Time series cross-validation (walk-forward)
   - Multiple error metrics
   - Residual analysis
   - Forecast confidence intervals

3. **Model Selection**
   - AIC/BIC criteria
   - Parameter tuning
   - Hyperparameter optimization
   - Ensemble approaches

4. **Practical Application**
   - Real dataset forecasting
   - Multi-step ahead prediction
   - Seasonal pattern handling

#### Key Concepts
- Statistical modeling of time series
- Parameter estimation
- Model diagnostics
- Practical forecasting workflow

#### Libraries Used
`statsmodels`, `pandas`, `numpy`, `matplotlib`, `prophet`

#### Output/Results
- Comparative performance table
- Model accuracy metrics
- Residual diagnostics
- Forecast confidence bands
- Model recommendation

---

### 6.3 M5 Forecasting Challenge: Neural Networks
**File**: `M5_Forecasting_NN_Comparison.ipynb`

#### Objectives
- Apply neural networks to time series
- Handle large-scale forecasting problems
- Compare NN with classical methods
- Implement ensemble strategies

#### Topics Covered
1. **Deep Learning for Time Series**
   - LSTM networks
   - GRU networks
   - Attention mechanisms
   - Conv1D for temporal patterns

2. **Data Preparation**
   - Windowing/sliding windows
   - Multivariate forecasting
   - External features
   - Normalization strategies

3. **M5 Challenge Details**
   - Store sales forecasting
   - 30,000+ product-store combinations
   - Hierarchical structure
   - Multiple forecasting horizons

4. **Advanced Techniques**
   - Hierarchical forecasting
   - Ensemble methods
   - Transfer learning
   - Multi-task learning

5. **Deployment Considerations**
   - Model scalability
   - Inference time
   - Memory efficiency
   - Batch predictions

#### Key Concepts
- Neural networks for sequences
- Multi-step prediction
- Large-scale forecasting
- Ensemble learning
- Practical ML deployment

#### Libraries Used
`tensorflow`, `keras`, `pandas`, `numpy`, `sklearn`, `matplotlib`

#### Output/Results
- NN model performance vs baselines
- Multi-model ensemble results
- Forecast accuracy by product
- Hierarchy level analysis
- Computational efficiency metrics

---

## 🎓 Learning Paths

### Path 1: Beginner ML
1. Livable_1 → Livrable_2 → Livrable_3

### Path 2: Deep Learning Specialist
1. Livable_1 → Livrable_3 → Livrable_4 → Livrable_5

### Path 3: NLP Enthusiast
1. Livable_1 → Livrable_2 → Livrable_6

### Path 4: Time Series Expert
1. Livable_1 → Livrable_5 (all notebooks)

---

## 📊 Common Metrics Summary

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MAE** | Average absolute errors | Regression |
| **RMSE** | Square root of mean squared errors | Regression |
| **R²** | Proportion of variance explained | Regression |
| **Accuracy** | Correct predictions / Total | Classification |
| **Precision** | TP / (TP + FP) | Classification |
| **Recall** | TP / (TP + FN) | Classification |
| **F1-Score** | Harmonic mean of Precision & Recall | Classification |
| **MAPE** | Mean Absolute Percentage Error | Forecasting |

---

## 🔧 Tools & Frameworks

| Tool | Purpose | Notebooks |
|------|---------|-----------|
| TensorFlow/Keras | Deep Learning | All |
| Scikit-Learn | ML Algorithms | 1, 4, 6 |
| Statsmodels | Time Series | 6 |
| NumPy | Numerical Computing | All |
| Pandas | Data Manipulation | All |
| Matplotlib | Visualization | All |
| Seaborn | Statistical Plots | 1, 2, 4, 6 |

---

## 📈 Progression Complexity

```
Complexity →
│
│ Livrable_6 (NLP)
│ Livrable_5 (Time Series Advanced)
│     │
│     ├─ M5 Forecasting
│     ├─ Model Comparison
│     └─ Time Series Basics
│
│ Livrable_4 (CNN Application)
│ Livrable_3 (CNN Fundamentals)
│ Livrable_2 (Neural Networks)
│ Livable_1 (Linear Regression)
└─────────────────────→ Topics
```

---

## 🚀 Next Steps After Learning

1. **Kaggle Competitions**: Apply skills to real challenges
2. **Personal Projects**: Build domain-specific models
3. **Research Papers**: Read latest DL research
4. **Advanced Frameworks**: PyTorch, JAX, etc.
5. **Production Deployment**: Learn MLOps and serving

---

**Last Updated**: December 2024
