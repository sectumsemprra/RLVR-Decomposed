# Google Colab Quick Start

## 🚀 Copy-Paste Setup (30 seconds)

Open a new Google Colab notebook and paste these commands:

### Step 1: Clone with Colab Files
```python
# Clone the repo and checkout the Colab-ready branch
!git clone https://github.com/sectumsemprra/RLVR-Decomposed.git
%cd RLVR-Decomposed
!git checkout claude/check-qwen3-dependencies-W4BzB

# Verify files
!ls -la requirements_colab.txt run_qwen3-4b_psr_nsr_colab_*.sh
```

### Step 2: Check Your GPU
```python
!nvidia-smi
```

### Step 3: Install Dependencies
```python
# Install packages (takes 5-10 minutes)
!pip install -r requirements_colab.txt

# Optional: Install flash-attention for better performance (takes 10-15 min)
# !python -m pip install flash-attn --no-build-isolation
```

### Step 4: Verify Installation
```python
import transformers, tensordict, ray, vllm
print(f"Transformers: {transformers.__version__}")
print(f"vLLM: {vllm.__version__}")
print(f"TensorDict: {tensordict.__version__}")
print(f"Ray: {ray.__version__}")
```

### Step 5: Setup Data
⚠️ **You need to provide your data files first!**

```python
# Create data directories
!mkdir -p data/math data/aime2025 data/amc23

# TODO: Add your data download commands here
# Example:
# !wget -O data/math/train.parquet YOUR_URL
# !wget -O data/math/test.parquet YOUR_URL
# !wget -O data/aime2025/test.parquet YOUR_URL
# !wget -O data/amc23/test.parquet YOUR_URL
```

### Step 6: Run Training with Real-Time Progress

**For Colab FREE (T4 GPU) - Use smaller model:**
```python
# Modify script to use smaller model (recommended for T4)
!sed -i 's/model_name=Qwen\/Qwen3-4B/model_name=Qwen\/Qwen2.5-0.5B-Instruct/' run_qwen3-4b_psr_nsr_colab_free.sh

# Run training with REAL-TIME OUTPUT (you'll see every epoch!)
!bash run_qwen3-4b_psr_nsr_colab_free.sh 2>&1 | tee training.log
```

**For Colab PRO (A100/V100) - Keep original model:**
```python
# Run with real-time output
!bash run_qwen3-4b_psr_nsr_colab_pro.sh 2>&1 | tee training.log
```

**You will see output like:**
```
Epoch 1/20 - Iteration 10/235 - Loss: 1.234 - ETA: 1.2 hours
Epoch 1/20 - Iteration 20/235 - Loss: 1.156 - ETA: 1.1 hours
...
Epoch 1 complete! Time: 1.3 hours
Epoch 2/20 - Starting...
```

### Step 7: Monitor Progress (Optional)

**In a separate cell, monitor training while it runs:**
```python
# Watch the last 20 lines of output, refreshing every 10 seconds
import time
from IPython.display import clear_output

for i in range(1000):
    clear_output(wait=True)
    !tail -n 20 training.log
    print(f"\n🔄 Refreshed {i+1} times | Press ■ (stop) to exit")
    time.sleep(10)
```

**Check current epoch:**
```python
!grep "Epoch" training.log | tail -5
```

**Check GPU usage:**
```python
!nvidia-smi
```

---

## 📓 Alternative: Use Pre-made Notebook

Instead of copy-pasting, you can:

1. Download `RLVR_Colab_Setup.ipynb` from the repo
2. Upload it to Google Colab
3. Run cells in order

---

## ⚠️ Common Issues

### "requirements_colab.txt not found"
You forgot to checkout the branch:
```python
!git checkout claude/check-qwen3-dependencies-W4BzB
```

### Out of Memory
Use smaller model:
```python
# Edit the script and change:
# model_name=Qwen/Qwen2.5-0.5B-Instruct
```

### Version Conflicts
Force reinstall:
```python
!pip uninstall -y vllm transformers tensordict
!pip install vllm==0.8.5 transformers==4.52.2 tensordict==0.7.2
```

---

## 📊 What to Expect

### Colab Free (T4, ~15GB)
- Model: Qwen2.5-0.5B or 1.5B recommended
- Speed: ~2-4 hours per epoch
- Quality: Lower but functional

### Colab Pro (A100, ~40GB)
- Model: Qwen3-4B works well
- Speed: ~30-60 minutes per epoch
- Quality: Much better

---

## 📚 Full Guide

For detailed troubleshooting, see:
```python
!cat COLAB_SETUP_GUIDE.md
```

---

**Good luck! 🚀**
