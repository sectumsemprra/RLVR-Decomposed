# Training Time Estimates for RLVR-Decomposed

## ⏱️ **Quick Answer**

| Setup | Model | Per Epoch | 20 Epochs Total | Feasible? |
|-------|-------|-----------|-----------------|-----------|
| **Original (8x A100)** | Qwen3-4B | 10-30 min | 3-10 hours | ✅ Ideal |
| **Colab Pro (1x A100)** | Qwen3-4B | 30-60 min | 10-20 hours | ✅ Good |
| **Colab Pro (1x A100)** | Qwen2.5-1.5B | 15-30 min | 5-10 hours | ✅ Better |
| **Colab Free (1x T4)** | Qwen3-4B | 3-5 hours | 60-100 hours | ⚠️ Challenging |
| **Colab Free (1x T4)** | Qwen2.5-1.5B | 1.5-3 hours | 30-60 hours | ⚠️ Doable |
| **Colab Free (1x T4)** | Qwen2.5-0.5B | 1-2 hours | 20-40 hours | ✅ Recommended |

---

## 📊 **Detailed Breakdown**

### Original Setup (8x A100 40GB GPUs)
**Configuration:**
- Batch size: 1024
- Mini batch: 256
- 8 GPUs with tensor parallelism
- Max response: 31,744 tokens

**Time Estimates:**
- Per epoch: **10-30 minutes**
- 20 epochs: **3-10 hours**
- Wall clock time: **Half a day**

---

### Colab Pro/Pro+ (1x A100 40GB)

#### With Qwen3-4B (Original Model)
**Configuration:**
- Batch size: 256 (4x smaller)
- Mini batch: 64
- Single GPU (no parallelism)
- Max response: 8,192 tokens

**Time Estimates:**
- Per epoch: **30-60 minutes**
- 20 epochs: **10-20 hours**
- Wall clock time: **~1 day**

**Colab Pro Limits:**
- Runtime: 24 hours max
- ✅ You can complete training in one session
- 💡 Save checkpoints every 7 epochs (default) as backup

#### With Qwen2.5-1.5B (Smaller Model)
**Time Estimates:**
- Per epoch: **15-30 minutes**
- 20 epochs: **5-10 hours**
- Wall clock time: **~Half day**
- ✅ Much more comfortable margin

#### With Qwen2.5-0.5B (Smallest Model)
**Time Estimates:**
- Per epoch: **8-15 minutes**
- 20 epochs: **3-5 hours**
- Wall clock time: **~Quarter day**
- ✅ Very fast, great for testing

---

### Colab Free (1x T4 16GB)

#### With Qwen3-4B (Original Model - NOT RECOMMENDED)
**Configuration:**
- Batch size: 32 (32x smaller)
- Mini batch: 8
- CPU offloading enabled
- Max response: 2,048 tokens

**Time Estimates:**
- Per epoch: **3-5 hours**
- 20 epochs: **60-100 hours**
- Wall clock time: **3-4 days**

**Major Issues:**
- ❌ Colab Free has 12-hour runtime limit
- ❌ Will disconnect 5-8 times during training
- ❌ High risk of OOM errors
- ❌ Very slow progress

#### With Qwen2.5-1.5B (Medium Model - ACCEPTABLE)
**Time Estimates:**
- Per epoch: **1.5-3 hours**
- 20 epochs: **30-60 hours**
- Wall clock time: **2-3 days**

**Issues:**
- ⚠️ Still needs 3-5 sessions due to 12-hour limit
- ⚠️ Need to save and resume from checkpoints
- ✓ More stable, less OOM risk

#### With Qwen2.5-0.5B (Smallest Model - RECOMMENDED)
**Time Estimates:**
- Per epoch: **1-2 hours**
- 20 epochs: **20-40 hours**
- Wall clock time: **2-3 days**

**Advantages:**
- ✅ Most stable on T4
- ✅ Lower OOM risk
- ✅ Can do ~6-10 epochs per session
- ⚠️ Still needs 2-4 sessions to complete

---

## 🔢 **Why These Differences?**

### Factors Affecting Training Speed:

1. **Number of GPUs**: 8 → 1 = 8x slower
2. **Batch Size**: 1024 → 32 = 32x more iterations
3. **Token Length**: 31744 → 2048 = Less computation per sample
4. **CPU Offloading**: Enabled on free tier = 20-30% slower
5. **Model Size**: 4B → 0.5B = 8x fewer parameters
6. **GPU Speed**: A100 vs T4 = 3-4x difference

### Combined Effect:
- **Colab Free vs Original**: 25-40x slower
- **Colab Pro vs Original**: 5-10x slower

---

## 💡 **Practical Recommendations**

### For Colab Free (T4):
1. **Use Qwen2.5-0.5B-Instruct** for stability
2. **Expect 2-4 training sessions** (with disconnects)
3. **Set up checkpoint saving** to Google Drive
4. **Reduce epochs to 10-15** if you want faster results
5. **Monitor closely** for OOM errors
6. **Run overnight** to maximize session time

### For Colab Pro (A100):
1. **Qwen3-4B works well** but Qwen2.5-1.5B is faster
2. **Can complete in 1 session** (stay under 24 hours)
3. **Still save checkpoints** every 7 epochs as backup
4. **Consider increasing batch size** if you want faster convergence
5. **Can train multiple models** in one day

---

## 📈 **Dataset Size Impact**

The MATH dataset has approximately:
- **Training samples**: ~7,500 problems
- **Test samples**: ~5,000 problems

**Iterations per epoch:**
- Original (batch=1024): ~7-8 iterations
- Colab Pro (batch=256): ~30 iterations
- Colab Free (batch=32): ~235 iterations

More iterations = more time, but also more stable gradients with small batches.

---

## 🔄 **Handling Colab Disconnects**

### Automatic Checkpoint Saving
The scripts save every 7 epochs by default:
```bash
trainer.save_freq=7
```

### Resume from Checkpoint
To resume training, you need to:
1. Mount Google Drive and copy checkpoints back
2. Add checkpoint path to the training command
3. Continue from where you left off

### Prevent Disconnects (Colab Free)
Run this in browser console to keep session alive:
```javascript
function ClickConnect(){
  console.log("Keep alive");
  document.querySelector("colab-connect-button")?.click();
}
setInterval(ClickConnect, 60000);
```

**Note:** This doesn't extend the 12-hour limit, just prevents idle disconnects.

---

## 🎯 **Reducing Training Time**

### Option 1: Reduce Epochs
```bash
# In the script, change:
trainer.total_epochs=10  # Instead of 20
```
- Time: Cut in half
- Quality: Still usable, may need more tuning

### Option 2: Smaller Model
```bash
model_name=Qwen/Qwen2.5-0.5B-Instruct  # 8x faster
```
- Time: Much faster
- Quality: Lower but good for testing

### Option 3: Reduce Validation Frequency
```bash
trainer.test_freq=14  # Instead of 7
trainer.save_freq=10  # Instead of 7
```
- Saves time on validation
- Less frequent checkpoints (trade-off)

### Option 4: Increase Batch Size (if memory allows)
```bash
# For Colab Pro, try:
data.train_batch_size=512  # Instead of 256
```
- Fewer iterations per epoch
- May need to adjust learning rate

---

## 📊 **Example Timeline**

### Colab Free with Qwen2.5-0.5B (Recommended):
```
Day 1: Session 1 (12 hours) → Epochs 1-7 ✓
       Save checkpoint to Drive

Day 2: Session 2 (12 hours) → Epochs 8-14 ✓
       Save checkpoint to Drive

Day 3: Session 3 (8 hours)  → Epochs 15-20 ✓
       Training complete!
```

### Colab Pro with Qwen3-4B:
```
Day 1: Single Session (15 hours) → All 20 epochs ✓
       Training complete!
```

---

## ⚡ **Speed Optimization Tips**

1. **Skip flash-attention installation** if in a hurry (saves 15 min setup time)
2. **Disable wandb** if you don't need experiment tracking
3. **Reduce validation frequency** to speed up training
4. **Use smaller max_response_length** if your problems allow it
5. **Enable mixed precision** (already enabled via gradient checkpointing)

---

## 🎓 **Bottom Line**

**For learning/testing:**
- Colab Free + Qwen2.5-0.5B: **20-40 hours** (2-3 days)
- Patience required but totally doable

**For serious training:**
- Colab Pro + Qwen2.5-1.5B: **5-10 hours** (1 day)
- Much better experience

**For production/research:**
- 8x A100 + Qwen3-4B: **3-10 hours** (half day)
- Optimal setup from the paper

---

**TL;DR:**
- Free tier: Plan for **2-4 days** with checkpointing
- Pro tier: Plan for **1 day**, single session
- Be patient with free tier, or invest in Pro for smoother experience! 💪
