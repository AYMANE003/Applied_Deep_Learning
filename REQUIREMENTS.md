# Requirements & Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: ~2GB for dependencies

---

## Python Dependencies

### Core Machine Learning Libraries

```
numpy>=1.21.0              # Numerical computing
pandas>=1.3.0              # Data manipulation and analysis
scikit-learn>=1.0.0        # Machine learning algorithms
scipy>=1.7.0               # Scientific computing
```

### Deep Learning Frameworks

```
tensorflow>=2.10.0         # TensorFlow with Keras
# OR
torch>=1.10.0              # PyTorch (if alternative needed)
torchvision>=0.11.0        # Computer vision utilities
```

### Data & Visualization

```
matplotlib>=3.4.0          # Plotting and visualization
seaborn>=0.11.0            # Statistical data visualization
plotly>=5.0.0              # Interactive visualizations
opencv-python>=4.5.0       # Computer vision library
```

### Time Series Analysis

```
statsmodels>=0.13.0        # Statistical models and tests
pmdarima>=2.0.0            # ARIMA models
prophet>=1.1.0             # Facebook's Prophet for forecasting
```

### NLP & Text Processing

```
nltk>=3.6.0                # Natural Language Toolkit
textblob>=0.17.0           # Simplified NLP
gensim>=4.0.0              # Topic modeling
```

### Jupyter & Development

```
jupyter>=1.0.0             # Jupyter Notebook
jupyterlab>=3.0.0          # JupyterLab interface (recommended)
ipython>=7.0.0             # Enhanced Python shell
ipywidgets>=7.6.0          # Interactive widgets
```

### Utilities

```
python-dotenv>=0.19.0      # Environment variables
tqdm>=4.60.0               # Progress bars
requests>=2.26.0           # HTTP requests
```

---

## Installation Methods

### Method 1: Using requirements.txt (Recommended)

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

4. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Method 2: Manual Installation

If you have a custom Python environment, install packages individually:

```bash
# Core libraries
pip install numpy pandas scikit-learn scipy

# Deep Learning
pip install tensorflow

# Visualization
pip install matplotlib seaborn plotly

# Time Series
pip install statsmodels pmdarima prophet

# NLP
pip install nltk

# Jupyter
pip install jupyter jupyterlab

# Development tools
pip install ipython ipywidgets
```

### Method 3: Using Anaconda (Alternative)

```bash
# Create conda environment
conda create -n deep_learning python=3.10

# Activate environment
conda activate deep_learning

# Install with conda
conda install numpy pandas scikit-learn tensorflow matplotlib jupyter
```

---

## Verification

After installation, verify everything is working:

### Check Python Version
```bash
python --version
# Output should be 3.8 or higher
```

### Check Key Packages
```python
python -c "
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('Scikit-learn:', sklearn.__version__)
print('TensorFlow:', tf.__version__)
print('Matplotlib:', matplotlib.__version__)
print('✅ All packages installed successfully!')
"
```

### Quick Test Script
Create and run `test_installation.py`:

```python
#!/usr/bin/env python
"""Test installation of all required packages."""

packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-Learn',
    'scipy': 'SciPy',
    'tensorflow': 'TensorFlow',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'statsmodels': 'Statsmodels',
    'nltk': 'NLTK',
    'jupyter': 'Jupyter',
}

print("Testing package installations...\n")

all_ok = True
for package, name in packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name:20} v{version}")
    except ImportError:
        print(f"❌ {name:20} NOT INSTALLED")
        all_ok = False

print("\n" + "="*50)
if all_ok:
    print("✅ All packages installed successfully!")
else:
    print("❌ Some packages are missing. Please install them.")
    print("\nRun: pip install -r requirements.txt")
```

Run it with:
```bash
python test_installation.py
```

---

## Optional Dependencies

### GPU Support (Recommended for faster training)

#### CUDA & cuDNN Installation
- **For NVIDIA GPUs**: Install CUDA and cuDNN
- Follow [TensorFlow GPU Setup Guide](https://www.tensorflow.org/install/gpu)

#### Verify GPU Setup
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should show available GPU devices
```

### Jupyter Extensions (Optional)

```bash
# Code autocompletion
pip install jupyter_contrib_nbextensions

# Dark theme
pip install jupyterthemes
jt -t monokai -f fira -fs 12

# Table of contents
jupyter nbextension enable toc2/main
```

---

## Troubleshooting

### Issue: "Module not found" error

**Solution**: Ensure your virtual environment is activated and packages are installed.

```bash
# Check active environment
which python  # macOS/Linux
where python  # Windows

# Reinstall package
pip install --force-reinstall package_name
```

### Issue: TensorFlow GPU not detected

**Solution**: Verify CUDA installation and GPU compatibility.

```bash
# Check GPU
nvidia-smi

# Verify TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

### Issue: Jupyter not launching

**Solution**: Reinstall Jupyter and check port availability.

```bash
# Reinstall
pip install --force-reinstall jupyter

# Use different port
jupyter notebook --port 8889
```

### Issue: Package version conflicts

**Solution**: Create a fresh virtual environment.

```bash
# Remove old environment
rm -rf venv  # macOS/Linux
rmdir venv /s  # Windows

# Recreate and reinstall
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Performance Optimization

### Recommended Settings

1. **Use Python 3.10** for better performance
2. **Install with pip using cached wheels** for faster installation
3. **Use GPU** for training if available
4. **Set number of threads** for parallel processing:
   ```python
   import os
   os.environ['OMP_NUM_THREADS'] = '4'
   ```

### Memory Management

For large datasets, limit TensorFlow memory usage:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

---

## Environment Variables (Optional)

Create a `.env` file in the project root:

```
PYTHONUNBUFFERED=1
TF_CPP_MIN_LOG_LEVEL=2
OMP_NUM_THREADS=4
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Docker Setup (Advanced)

For containerized setup, create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
```

Build and run:
```bash
docker build -t deep_learning .
docker run -p 8888:8888 deep_learning
```

---

## Getting Help

1. **Check official documentation**:
   - [TensorFlow Docs](https://www.tensorflow.org/)
   - [Scikit-Learn Docs](https://scikit-learn.org/)
   - [Pandas Docs](https://pandas.pydata.org/)

2. **Search Stack Overflow** for your error message

3. **Check GitHub Issues** in this repository

4. **Visit forums**:
   - [TensorFlow Community](https://discuss.tensorflow.org/)
   - [Kaggle Discussions](https://www.kaggle.com/discussion)

---

## Next Steps

After successful installation:

1. ✅ Verify all packages with `test_installation.py`
2. 📖 Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for quick start
3. 🚀 Launch Jupyter and open first notebook
4. 📚 Follow the learning path in [README.md](README.md)

---

**Happy learning! If you encounter issues, refer to the Troubleshooting section above.**
