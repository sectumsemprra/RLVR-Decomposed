# Real-Time Training Monitoring Guide

## 🖥️ Seeing Training Progress in Colab Terminal

When you run the training script in Colab, you'll see live output showing epoch progress, loss values, and timing information.

---

## ✅ **Method 1: Direct Execution (Recommended)**

This shows **real-time output** directly in your Colab cell:

```python
# Use ! for direct execution with live output
!bash run_qwen3-4b_psr_nsr_colab_free.sh
```

**You will see output like:**
```
Starting Ray instance...
Loading model Qwen/Qwen2.5-0.5B-Instruct...
Starting training...

Epoch 1/20
==================================================
Step 1/235 | Loss: 2.456 | Time: 12.3s
Step 10/235 | Loss: 2.234 | Time: 8.7s
Step 20/235 | Loss: 2.112 | Time: 8.5s
...
Step 235/235 | Loss: 1.876 | Time: 8.2s
Epoch 1 complete | Avg Loss: 1.923 | Time: 1.2 hours
Validation: Accuracy 45.2%

Epoch 2/20
==================================================
Step 1/235 | Loss: 1.834 | Time: 8.1s
...
```

---

## ✅ **Method 2: Unbuffered Output (If Method 1 is Slow)**

Sometimes Python buffers output. Force unbuffered output:

```python
# Force unbuffered output for real-time viewing
!python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=positive \
    ...  # (rest of parameters from the script)
```

Or modify the script to use unbuffered Python:
```bash
# Add -u flag to python in the script
python3 -u -m verl.trainer.main_ppo \
```

---

## ✅ **Method 3: Add Progress Monitoring**

Run training in background and monitor logs:

```python
# Start training in background
!bash run_qwen3-4b_psr_nsr_colab_free.sh > training.log 2>&1 &

# Monitor in real-time (run in another cell)
!tail -f training.log
```

---

## 📊 **What You'll See During Training**

### **Stage 1: Initialization (2-5 minutes)**
```
RAY: Starting Ray cluster...
RAY: Successfully started local Ray instance
Loading model from Qwen/Qwen2.5-0.5B-Instruct
Downloading model files... (if first run)
Model loaded successfully
Initializing trainer...
Creating dataloaders...
```

### **Stage 2: Epoch Training (1-2 hours per epoch)**
```
==================== Epoch 1/20 ====================
Train batch size: 32
Number of iterations: 235
Starting rollout generation...

Iteration 1/235
  Rollout: Generating 8 samples... Done (15.2s)
  Actor forward: Computing advantages... Done (3.4s)
  Critic update: Loss 0.234 (2.1s)
  Actor update: Policy loss 1.234, Value loss 0.456 (4.3s)
  Total iteration time: 25.0s
  ETA: 1.6 hours

Iteration 10/235
  Rollout: Generating 8 samples... Done (8.3s)
  Actor forward: Computing advantages... Done (2.1s)
  Critic update: Loss 0.198 (1.8s)
  Actor update: Policy loss 1.156, Value loss 0.421 (3.9s)
  Total iteration time: 16.1s
  ETA: 1.2 hours

...

Iteration 235/235
  Rollout: Generating 8 samples... Done (7.9s)
  Actor forward: Computing advantages... Done (2.0s)
  Critic update: Loss 0.145 (1.7s)
  Actor update: Policy loss 0.876, Value loss 0.312 (3.8s)
  Total iteration time: 15.4s

Epoch 1 Summary:
  Total time: 1.3 hours
  Average policy loss: 1.023
  Average value loss: 0.378
  Average reward: 0.234

Saving checkpoint to ./checkpoints/epoch_1/
```

### **Stage 3: Validation (2-5 minutes)**
```
Running validation on test sets...
  MATH test: 456/1000 correct (45.6%)
  AIME2025: 12/30 correct (40.0%)
  AMC23: 23/40 correct (57.5%)

Overall validation accuracy: 46.1%
```

### **Stage 4: Next Epoch**
```
==================== Epoch 2/20 ====================
...
```

---

## 🔍 **Monitor Specific Metrics**

### **Check Current Epoch:**
```python
# In a separate cell while training runs
!grep -a "Epoch" training.log | tail -1
```
Output: `Epoch 5/20`

### **Check Latest Loss:**
```python
!grep -a "policy loss" training.log | tail -5
```

### **Check ETA (Estimated Time):**
```python
!grep -a "ETA" training.log | tail -1
```

### **Check if Training is Still Running:**
```python
!ps aux | grep "verl.trainer.main_ppo"
```

---

## 📈 **Watch Progress in Real-Time**

### **Option A: Live Tail**
```python
# Run this in a separate cell
!tail -f -n 50 training.log
# Press stop button when you want to stop watching
```

### **Option B: Auto-Refresh Cell**
```python
import time
from IPython.display import clear_output

# Run this cell to auto-refresh progress every 30 seconds
for i in range(1000):  # Run for many iterations
    clear_output(wait=True)
    !tail -n 30 training.log
    print(f"\n🔄 Auto-refreshing... (iteration {i+1})")
    print("(Stop this cell when done)")
    time.sleep(30)  # Wait 30 seconds before refresh
```

### **Option C: Progress Bar (if available)**
If the training script uses tqdm or similar:
```
Epoch 1/20: 100%|██████████| 235/235 [1:23:45<00:00, 21.3s/it]
Policy Loss: 1.023 | Value Loss: 0.378 | Reward: 0.234
```

---

## 🎯 **Key Metrics to Watch**

### **1. Epoch Number**
Shows which epoch you're on: `Epoch 5/20`

### **2. Iteration Progress**
Shows steps within epoch: `Iteration 100/235`

### **3. Time Estimates**
- `ETA: 1.2 hours` - Time remaining for current epoch
- `Total time: 1.3 hours` - Time taken for completed epoch

### **4. Loss Values**
- **Policy Loss** (should decrease): `1.234 → 0.876`
- **Value Loss** (should decrease): `0.456 → 0.312`

### **5. Rewards**
- **Average Reward** (should increase): `0.234 → 0.456`

### **6. Validation Accuracy**
- Test accuracy after each epoch: `45.6% → 48.3% → 51.2%`

---

## 🚨 **What to Watch For**

### **Good Signs ✅**
- Loss values decreasing over time
- Rewards increasing over time
- Regular "Epoch X/20" messages
- Checkpoint saves every 7 epochs

### **Warning Signs ⚠️**
- No output for 10+ minutes (might be stuck)
- `CUDA out of memory` errors (need to reduce batch size)
- Loss values = NaN (training diverged)
- Same epoch for too long (frozen process)

---

## 🛠️ **Troubleshooting Output Issues**

### **Problem: No output showing**
```python
# Make sure you're not using & (background) without tailing
# Use this instead:
!bash run_qwen3-4b_psr_nsr_colab_free.sh 2>&1 | tee training.log
```

### **Problem: Output is buffered (delayed)**
```python
# Force flush with stdbuf
!stdbuf -oL -eL bash run_qwen3-4b_psr_nsr_colab_free.sh
```

### **Problem: Want to see GPU usage**
```python
# In a separate cell
!nvidia-smi
# Or watch continuously
!watch -n 5 nvidia-smi  # Updates every 5 seconds
```

---

## 💾 **Save Output to File**

### **Save everything to log file:**
```python
!bash run_qwen3-4b_psr_nsr_colab_free.sh 2>&1 | tee training.log
```
- You'll see output in real-time
- Everything is also saved to `training.log`
- Can review later or download

### **Download log file:**
```python
from google.colab import files
files.download('training.log')
```

---

## 📱 **Get Notifications**

### **Email notification when epoch completes:**
Add to the training script (advanced):
```python
# After each epoch, send email
import smtplib
# ... (email code)
```

### **Simple: Print loud completion message:**
```bash
# At end of script, add:
echo "🎉🎉🎉 TRAINING COMPLETE! 🎉🎉🎉"
echo "Check results in ./checkpoints/"
```

---

## 🔧 **Quick Reference Commands**

```python
# Start training with real-time output
!bash run_qwen3-4b_psr_nsr_colab_free.sh 2>&1 | tee training.log

# In another cell - monitor progress
!tail -f training.log

# Check current epoch
!grep "Epoch" training.log | tail -1

# Check GPU usage
!nvidia-smi

# Check if process is running
!ps aux | grep verl

# Kill if needed
!pkill -f verl.trainer.main_ppo
```

---

## 📊 **Expected Output Timeline**

```
Minute 0-5:   Initialization, model loading
Minute 5-90:  Epoch 1 training (235 iterations)
Minute 90-92: Validation
Minute 92-95: Checkpoint save

Minute 95-170:  Epoch 2 training
Minute 170-172: Validation
... (repeat)

Hour 20-40: Final epoch, training complete! 🎉
```

---

## ✅ **Simple Copy-Paste Solution**

Just run this in Colab:

```python
# This will show ALL output in real-time
!bash run_qwen3-4b_psr_nsr_colab_free.sh 2>&1 | tee training.log
```

You'll see every epoch, every iteration, and all progress live in your Colab cell! 🚀
