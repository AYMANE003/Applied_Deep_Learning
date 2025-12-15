# Quick Setup Guide

**Get started in 5 minutes!**

---

## Step 1: Clone & Navigate

```bash
git clone https://github.com/yourusername/Applied_Deep_Learning.git
cd Applied_Deep_Learning
```

---

## Step 2: Create Virtual Environment

```bash
# Create
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected time**: 2-5 minutes depending on internet speed

---

## Step 4: Launch Jupyter

```bash
jupyter notebook
```

or for a better interface:

```bash
jupyter lab
```

Your browser will open automatically. If not, visit: **http://localhost:8888**

---

## Step 5: Open Your First Notebook

1. Navigate to the root folder
2. Click on any `.ipynb` file to open it
3. **Recommended first notebook**: `Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb`

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed successfully
- [ ] Jupyter running in browser
- [ ] First notebook opens without errors

---

## 🚀 First Steps in a Notebook

### Running Cells

1. Click on a code cell (gray box with code)
2. Press **Shift + Enter** to run
3. Results appear below the cell
4. Continue to next cell

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | **Shift + Enter** |
| Create new cell below | **B** |
| Create new cell above | **A** |
| Delete cell | **DD** |
| Cut cell | **X** |
| Paste cell | **V** |
| Change to Markdown | **M** |
| Change to Code | **Y** |

---

## 📊 Recommended Learning Order

### Day 1: Foundations
```
1. Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb
   ↓
   Learn: Linear regression, gradient descent, model training
```

### Day 2: Neural Networks
```
2. Livrable_2_RNA_(RL+ReLu_Sigmoid).ipynb
   ↓
   Learn: ANN basics, activation functions, classification
```

### Day 3: Computer Vision - Part 1
```
3. Livrable_3_CNN_solo.ipynb
   ↓
   Learn: Convolution operation, filters, feature extraction
```

### Day 4: Computer Vision - Part 2
```
4. Livrable_4_CNN_Mnist_Dataset.ipynb
   ↓
   Learn: Image classification, CNN training, evaluation
```

### Day 5: Advanced Topics
```
5. Livrable_6_Sarcasm_NLP.ipynb
   ↓
   Learn: NLP, text processing, deep learning for sequences
```

### Days 6-7: Time Series
```
6. Livrable_5_Time_Series/ (All notebooks)
   ↓
   Learn: Time series analysis, forecasting, advanced models
```

---

## 🎯 Tips & Tricks

### Save Your Work
- Notebooks auto-save, but manually save with **Ctrl/Cmd + S**
- Always commit to git before closing

### Debugging
- **Cell Error?** Look at the red error message
- **Check imports** - Run first cell that has imports
- **Restart kernel** if variables are undefined: Kernel → Restart

### Performance
- **Slow notebook?** Try restarting kernel and running from top
- **Out of memory?** Close unused notebooks and browser tabs
- **GPU not used?** Check REQUIREMENTS.md for GPU setup

### Modification
- Don't modify input notebooks - make copies instead
- Create personal branch for experiments: `git checkout -b my-experiments`
- Document your changes and findings

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
# Solution: Reinstall in active environment
pip install tensorflow
```

### Issue: Jupyter not opening
```bash
# Solution: Check if port is in use
jupyter notebook --port 8889
```

### Issue: Cell output is too long
```python
# In Jupyter, add to first cell:
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
```

### Issue: Memory errors with large datasets
```python
# Reduce data size or use generators
import gc
gc.collect()  # Force garbage collection
```

---

## 📝 Working with Notebooks

### Creating Your Own Analysis

1. **Copy notebook**: Don't modify original files
   ```bash
   cp Livrable_4_CNN_Mnist_Dataset.ipynb my_cnn_experiments.ipynb
   ```

2. **Create markdown cells** to document findings
3. **Add comments** in code cells
4. **Create visualizations** to understand results
5. **Save and commit** to git

### Template for a Good notebook cell

```python
# ============================================================================
# TASK: [Clear description of what you're doing]
# ============================================================================

# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 2: Load or prepare data
# ... your code here ...

# Step 3: Process and visualize
# ... your code here ...

# RESULT: [What you discovered or achieved]
```

---

## 🔄 Git Workflow

### Save Your Progress

```bash
# Check what changed
git status

# Add changes
git add .

# Save with message
git commit -m "Completed CNN experiments on MNIST"

# Push to GitHub
git push origin main
```

### Collaborate

```bash
# Get latest changes
git pull origin main

# Create feature branch
git checkout -b feature/my-improvement

# After improvements, push and create Pull Request
```

---

## 📚 Resources During Learning

### While Working:
- Keep [REQUIREMENTS.md](REQUIREMENTS.md) open for package docs
- Use **Shift + Tab** in Jupyter to see function documentation
- Type `?function_name` to see help

### External Resources:
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Scikit-Learn Examples](https://scikit-learn.org/stable/auto_examples/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [GitHub Copilot](#) for code help

---

## ⚙️ Advanced: Jupyter Configuration

### Create Custom Jupyter Config

```bash
# Generate config
jupyter notebook --generate-config

# Edit config file:
# Windows: C:\Users\<username>\.jupyter\jupyter_notebook_config.py
# macOS: ~/.jupyter/jupyter_notebook_config.py
# Linux: ~/.jupyter/jupyter_notebook_config.py
```

### Useful Settings to Modify

```python
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = '/path/to/notebooks'
c.NotebookApp.port = 8888
```

---

## 🎓 How to Use These Notebooks Effectively

### Active Learning Strategy

1. **Read** the introduction and objectives
2. **Run** each cell one-by-one (don't skip cells!)
3. **Experiment** by modifying parameters
4. **Visualize** outputs and understand patterns
5. **Reflect** on what you learned
6. **Document** key findings

### Avoid These Mistakes

❌ Running all cells at once (Shift + Ctrl + Enter)
❌ Skipping markdown cells (they contain important info)
❌ Not understanding the code, just copy-pasting
❌ Modifying original notebooks (create copies!)

### Do This Instead

✅ Run cells sequentially
✅ Read explanations carefully
✅ Understand each line of code
✅ Experiment and modify parameters
✅ Create your own notebooks

---

## 🚀 What's Next?

After completing basics:

1. **Deep Dive**: Study additional resources for each topic
2. **Experiment**: Modify code and hyperparameters
3. **Kaggle**: Apply skills to competition datasets
4. **Project**: Build your own ML/DL project
5. **Share**: Contribute improvements back to this repo

---

## 📞 Need Help?

1. **Check REQUIREMENTS.md** for setup issues
2. **Read notebook markdown cells** - they contain explanations
3. **Search error messages** on Stack Overflow
4. **Check GitHub Issues** in this repository
5. **Ask in Discussions** tab

---

## ✨ You're Ready to Start!

```bash
# Activate environment
source venv/bin/activate  # or: venv\Scripts\activate

# Launch Jupyter
jupyter notebook

# Open: Livable_1_RL_from_Scratch_SickitLearn_TensorFlow.ipynb

# Press Shift + Enter to run first cell

# Happy Learning! 🚀
```

---

**Remember**: The best way to learn is by doing. Don't just read - code along and experiment!

---

*Last Updated: December 2024*
