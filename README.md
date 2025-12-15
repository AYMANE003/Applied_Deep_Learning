# Applied Deep Learning - Course Project Collection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📚 Overview

This repository contains a comprehensive collection of **Jupyter Notebook deliverables** from the **Applied Deep Learning** course. It showcases practical implementations of machine learning and deep learning concepts, from fundamental algorithms to advanced neural network architectures.

The project includes step-by-step explorations of:
- **Regression** (Linear, Neural Networks)
- **Artificial Neural Networks (RNA)** with different activation functions
- **Convolutional Neural Networks (CNN)**
- **Natural Language Processing (NLP)**
- **Time Series Forecasting**

---

## 📁 Project Structure

```
Applied_Deep_Learning/
│
├── README.md                                    # This file
├── REQUIREMENTS.md                              # Dependencies and installation
├── SETUP_GUIDE.md                               # Quick start guide
│
├── Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb
│   └── Linear Regression: From scratch, Scikit-Learn, and TensorFlow
│
├── Livrable_2_RNA_(RL+ReLu_Sigmoid).ipynb
│   └── Artificial Neural Networks: ReLU vs Sigmoid activation functions
│
├── Livrable_3_CNN_solo.ipynb
│   └── CNN Fundamentals: Convolution operations and filtering
│
├── Livrable_4_CNN_Mnist_Dataset.ipynb
│   └── CNN Application: MNIST digit classification
│
├── Livrable_6_Sarcasm_NLP.ipynb
│   └── Natural Language Processing: Sarcasm detection
│
└── Livrable_5_Time_Series/
    ├── Time_Series_Séance1.ipynb               # Session 1: Time series basics
    ├── Time_Series_Séance2_models_comparision.ipynb  # Session 2: Model comparison
    ├── M5_Forecasting_NN_Comparison.ipynb      # M5 Challenge: NN forecasting
    ├── m5-forecasting-accuracy-dataset.html
    └── models.h5.html
```

---

## 🎓 Notebook Details

### 1. **Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb**
**Linear Regression Implementation**

- **Topics**: Regression, Gradient Descent, Model Comparison
- **Methods**:
  - ✍️ From-scratch implementation with gradient descent
  - 🔧 Scikit-Learn LinearRegression
  - 🧠 TensorFlow/Keras implementation
- **Dataset**: House prices (synthetic)
- **Key Concepts**: 
  - Loss functions (MSE)
  - Model evaluation (R², RMSE)
  - Hyperparameter tuning

---

### 2. **Livrable_2_RNA_(RL+ReLu_Sigmoid).ipynb**
**Artificial Neural Networks - Activation Functions Study**

- **Topics**: ANN, Activation Functions, Classification
- **Focus**: Comparing ReLU vs Sigmoid activation
- **Datasets**:
  - Well-separated blobs (Gaussian)
  - Less separable clusters
  - Moon dataset (non-linear)
  - Helix dataset (complex spiral)
- **Key Concepts**:
  - Forward and backpropagation
  - Decision boundaries visualization
  - Non-linear separation
  - Performance metrics

---

### 3. **Livrable_3_CNN_solo.ipynb**
**Convolutional Neural Networks - Fundamentals**

- **Topics**: CNN Architecture, Convolution Operation, Filters
- **Configuration**:
  - Input: 5×5×3 RGB image
  - 2 filters (3×3×3)
  - Stride: 2
  - Padding: 1
- **Visualizations**: Feature maps, filter outputs
- **Key Concepts**:
  - Convolution operation
  - Stride and padding effects
  - Pooling operations
  - Feature extraction

---

### 4. **Livrable_4_CNN_Mnist_Dataset.ipynb**
**CNN Application - MNIST Digit Classification**

- **Topics**: Image Classification, Deep CNN, Data Preprocessing
- **Dataset**: MNIST (70,000 handwritten digits)
- **Architecture**: Multi-layer CNN with pooling
- **Key Concepts**:
  - Data normalization and augmentation
  - CNN training and validation
  - Confusion matrices
  - Model performance analysis
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

### 5. **Livrable_6_Sarcasm_NLP.ipynb**
**Natural Language Processing - Sarcasm Detection**

- **Topics**: NLP, Text Classification, Deep Learning for NLP
- **Dataset**: Sarcasm detection dataset (headlines)
- **Methods**:
  - Text preprocessing and tokenization
  - Embedding layers
  - LSTM/RNN architectures
  - Sentiment analysis
- **Key Concepts**:
  - Word embeddings
  - Sequence models
  - Attention mechanisms
  - Class imbalance handling

---

### 6. **Livrable_5_Time_Series/ (Folder)**
**Time Series Forecasting**

#### 6.1 **Time_Series_Séance1.ipynb**
- Introduction to time series analysis
- Stationarity tests (ADF)
- Decomposition (trend, seasonality)
- Basic forecasting methods (ARIMA, Exponential Smoothing)

#### 6.2 **Time_Series_Séance2_models_comparision.ipynb**
- Advanced models comparison
- ARIMA vs Prophet vs Neural Networks
- Cross-validation for time series
- Performance evaluation metrics

#### 6.3 **M5_Forecasting_NN_Comparison.ipynb**
- M5 Kaggle Challenge: Store sales forecasting
- Neural Network approaches for time series
- LSTM and GRU implementations
- Hierarchical forecasting
- Ensemble methods comparison

**Key Concepts**:
  - Temporal patterns recognition
  - Seasonality and trend analysis
  - Deep learning for sequences
  - Multi-step ahead forecasting

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- See [REQUIREMENTS.md](REQUIREMENTS.md) for dependencies

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Applied_Deep_Learning.git
   cd Applied_Deep_Learning
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

5. **Open any notebook** and start exploring!

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## 📊 Key Technologies & Libraries

| Technology | Version | Purpose |
|-----------|---------|---------|
| **TensorFlow/Keras** | 2.x | Deep Learning framework |
| **PyTorch** | Latest | Alternative DL framework (if used) |
| **Scikit-Learn** | Latest | Machine Learning algorithms |
| **Pandas** | Latest | Data manipulation |
| **NumPy** | Latest | Numerical computing |
| **Matplotlib** | Latest | Visualization |
| **Seaborn** | Latest | Statistical visualizations |
| **Statsmodels** | Latest | Time series analysis |

---

## 📈 Learning Path

### Beginner Level
1. Start with **Livable_1_RL_from_Scratch** - Understand ML fundamentals
2. Follow **Livrable_2_RNA** - Learn neural network concepts

### Intermediate Level
3. Study **Livrable_3_CNN_solo** - CNN architecture basics
4. Work through **Livrable_4_CNN_Mnist** - Practical CNN application

### Advanced Level
5. Explore **Livrable_6_Sarcasm_NLP** - Advanced NLP techniques
6. Deep dive into **Livrable_5_Time_Series** - Complex temporal patterns

---

## 🎯 Course Objectives Covered

- ✅ Linear Regression from first principles
- ✅ Artificial Neural Networks (ANN) architecture
- ✅ Activation functions and their impact
- ✅ Convolutional Neural Networks (CNN)
- ✅ Image classification tasks
- ✅ Natural Language Processing
- ✅ Time series forecasting
- ✅ Model evaluation and comparison
- ✅ Hyperparameter optimization
- ✅ Data preprocessing and augmentation

---

## 📝 Features of Each Notebook

### Common Elements
- 📖 **Theory Introduction**: Mathematical foundations explained
- 💻 **Code Implementation**: Well-commented Python code
- 📊 **Visualizations**: Charts and graphs for better understanding
- 📈 **Performance Metrics**: Quantitative evaluation
- 🧪 **Experiments**: Multiple test cases and scenarios
- 🎓 **Educational Notes**: Key takeaways and insights

---

## 🔧 Running Individual Notebooks

Each notebook is **self-contained** and can be run independently:

```bash
# For example, to run the MNIST CNN notebook:
jupyter notebook Livrable_4_CNN_Mnist_Dataset.ipynb
```

Make sure all required packages are installed beforehand.

---

## 💡 Tips & Best Practices

1. **Run cells sequentially** - Each notebook assumes top-to-bottom execution
2. **Check dependencies** - Refer to [REQUIREMENTS.md](REQUIREMENTS.md)
3. **Understand concepts** - Read markdown cells carefully
4. **Experiment** - Modify hyperparameters and observe results
5. **Visualize** - Pay attention to plots and interpretations
6. **Document** - Add your own notes and findings

---

## 🤝 Contributing

To contribute improvements or corrections:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-Learn User Guide](https://scikit-learn.org/)
- [Deep Learning Specialization (Andrew Ng)](https://www.deeplearning.ai/)
- [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer)
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_LFPM5gQa)

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 👤 Author

**Course**: Applied Deep Learning  
**Created**: 2024  
**Last Updated**: December 2024

---

## ⭐ Star & Fork

If you found this repository helpful, please consider:
- ⭐ Starring this repository
- 🍴 Forking for your own learning
- 📢 Sharing with fellow learners

---

## 📞 Support

For questions, issues, or suggestions:
- 📧 Create an GitHub Issue
- 💬 Start a Discussion
- 🐛 Report bugs with detailed information

---

**Happy Learning! 🚀**

*"The best way to learn machine learning is by doing." - Hands-on experimentation*
