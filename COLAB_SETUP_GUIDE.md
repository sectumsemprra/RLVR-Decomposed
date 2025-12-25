# Google Colab Setup Guide for RLVR-Decomposed

This guide will help you run the RLVR-Decomposed training on Google Colab (both Free and Pro tiers).

## ⚠️ Important Limitations

### Colab Free Tier (T4 GPU ~15GB VRAM)
- **Can run**: Yes, but with significant limitations
- **Model**: Recommend Qwen2.5-0.5B or Qwen2.5-1.5B (Qwen3-4B will be very tight)
- **Speed**: 20-30x slower than 8-GPU setup
- **Batch size**: Very small (32)
- **Max response**: 2048 tokens (vs 31744 in original)
- **Training time**: Several hours per epoch
- **May disconnect**: Free tier has 12-hour runtime limit

### Colab Pro/Pro+ (A100 40GB or V100 32GB)
- **Can run**: Yes, much better performance
- **Model**: Qwen3-4B works well
- **Speed**: 5-10x slower than 8-GPU setup
- **Batch size**: Moderate (256)
- **Max response**: 8192 tokens
- **Training time**: Faster, more stable

## 📦 Installation Instructions

### Step 1: Setup Python Environment in Colab

```python
# Run this in a Colab cell
!git clone https://github.com/YOUR_USERNAME/RLVR-Decomposed.git
%cd RLVR-Decomposed

# Check GPU type
!nvidia-smi
```

### Step 2: Install Dependencies

```python
# Install main dependencies
!pip install -r requirements_colab.txt

# Install flash-attention (optional but recommended)
# This may take 10-15 minutes
!python -m pip install flash-attn --no-build-isolation
```

**Note**: If flash-attention installation fails, you can skip it. The training will still work but may be slower.

### Step 3: Verify Installation

```python
# Check critical packages
import transformers
import vllm
import tensordict
import ray

print(f"Transformers: {transformers.__version__}")  # Should be 4.52.2
print(f"vLLM: {vllm.__version__}")  # Should be 0.8.5
print(f"TensorDict: {tensordict.__version__}")  # Should be 0.7.2
print(f"Ray: {ray.__version__}")  # Should be >= 2.10
```

### Step 4: Prepare Data

Make sure you have the required data files:
```bash
./data/math/train.parquet
./data/math/test.parquet
./data/aime2025/test.parquet
./data/amc23/test.parquet
```

If you don't have these files, you'll need to download or generate them according to the project's data preparation instructions.

### Step 5: Configure Weights & Biases (Optional)

```python
# Login to wandb for experiment tracking
import wandb
wandb.login()
```

## 🚀 Running Training

### For Colab Free Tier:

```bash
# Make script executable
!chmod +x run_qwen3-4b_psr_nsr_colab_free.sh

# Run training
!bash run_qwen3-4b_psr_nsr_colab_free.sh
```

**Recommended model change for Free tier** (edit the script first):
```bash
# Change this line in run_qwen3-4b_psr_nsr_colab_free.sh:
model_name=Qwen/Qwen2.5-0.5B-Instruct  # Much more stable on T4
```

### For Colab Pro/Pro+:

```bash
# Make script executable
!chmod +x run_qwen3-4b_psr_nsr_colab_pro.sh

# Run training
!bash run_qwen3-4b_psr_nsr_colab_pro.sh
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors, try these in order:

1. **Use a smaller model**:
   ```bash
   # Edit the script and change:
   model_name=Qwen/Qwen2.5-0.5B-Instruct
   ```

2. **Reduce batch size further**:
   ```bash
   data.train_batch_size=16  # or even 8
   ```

3. **Reduce token limits**:
   ```bash
   actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2000
   actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3000
   actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3000
   ```

4. **Reduce rollout samples**:
   ```bash
   actor_rollout_ref.rollout.n=2  # Instead of 4
   ```

### Script Freezes After "Started a local Ray instance"

Add this fix to the script:
```python
# Edit verl/trainer/main_ppo.py
# Find ray.init() and modify to:
ray.init(num_cpus=2, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
```

Or run in Colab:
```python
import ray
ray.init(num_cpus=2)
```

### Colab Disconnects During Training

**For Free Tier**:
- Use this JavaScript in browser console to prevent disconnection:
```javascript
function ClickConnect(){
  console.log("Clicking connect...");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
```

**Better Solution**: Use Colab Pro for longer runtimes (24 hours)

### Version Conflicts

If you encounter version conflicts:
```bash
# Uninstall conflicting packages
!pip uninstall -y vllm transformers tensordict

# Reinstall with exact versions
!pip install vllm==0.8.5 transformers==4.52.2 tensordict==0.7.2
```

## 📊 Monitoring Training

### Using Weights & Biases
- Go to https://wandb.ai
- Check your project: `verl-colab-free` or `verl-colab-pro`
- Monitor loss, rewards, and other metrics

### Using Colab Output
The script will print progress including:
- Epoch number
- Loss values
- Reward statistics
- GPU memory usage

## 💾 Saving Checkpoints

Checkpoints are saved every 7 epochs by default (`trainer.save_freq=7`).

To download checkpoints to your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints
!cp -r /path/to/checkpoints /content/drive/MyDrive/rlvr_checkpoints/
```

## 🎯 Training Variants

### PSR (Positive State Reinforcement) - Default
Already configured in the scripts

### NSR (Negative State Reinforcement)
Edit the script and change:
```bash
advantage="negative"   # Uncomment this
# advantage="positive"   # Comment this out
```

### W-REINFORCE (Weighted REINFORCE)
Edit the script and change:
```bash
advantage="weighted"   # Uncomment this
positive_advantage_weight=0.1   # Uncomment and add to command
```

Then add this line to the python command (before `$@`):
```bash
algorithm.positive_advantage_weight=$positive_advantage_weight \
```

## 📈 Expected Performance

### Colab Free (T4)
- **Time per epoch**: 2-4 hours (with small model)
- **Convergence**: Slower, may need more epochs
- **Final accuracy**: May be lower due to constraints

### Colab Pro (A100)
- **Time per epoch**: 30-60 minutes
- **Convergence**: Much better
- **Final accuracy**: Closer to published results

## ❓ FAQ

**Q: Can I use the original script on Colab?**
A: No, it requires 8 GPUs. Colab only provides 1 GPU.

**Q: Will the model quality be the same?**
A: No, smaller batches and shorter sequences will affect quality. But you can still see the training dynamics and test the approach.

**Q: How long will training take?**
A: Free tier: 40-80 hours for 20 epochs. Pro tier: 10-20 hours for 20 epochs.

**Q: Can I pause and resume training?**
A: Yes, checkpoints are saved. You can load from the last checkpoint.

**Q: What's the minimum GPU I need?**
A: T4 (15GB) for very small models, V100/A100 (32GB+) for Qwen3-4B.

## 📚 Additional Resources

- [VERL Documentation](https://verl.readthedocs.io/)
- [Original Paper](https://arxiv.org/abs/2506.01347)
- [Hugging Face Models](https://huggingface.co/collections/TianHongZXY/rlvr-decomposed-683c0cd7151b769d8ea5915c)

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the original README.md
3. Check GPU memory usage: `!nvidia-smi`
4. Verify package versions
5. Try with a smaller model first

---

**Good luck with your training! 🚀**
