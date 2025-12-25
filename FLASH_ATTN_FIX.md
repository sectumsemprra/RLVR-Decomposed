# Flash Attention Fix for Colab

## 🚨 Problem

If you see this error:
```
ModuleNotFoundError: No module named 'flash_attn'
```

Or Ray worker initialization failures mentioning flash_attn.

## ✅ Solution

The Colab training scripts (`run_qwen3-4b_psr_nsr_colab_free.sh` and `run_qwen3-4b_psr_nsr_colab_pro.sh`) have been **fixed** to work without flash-attention.

### What was changed:

1. **Added environment variable** to disable flash-attn:
   ```bash
   export VERL_DISABLE_FLASH_ATTN=1
   ```

2. **Added explicit attention implementation** parameter:
   ```bash
   actor_rollout_ref.model.attn_implementation=eager
   ```

3. **Updated requirements** - flash-attn is no longer mentioned as needed

## 📝 What This Means

- ✅ **No need to install flash-attn** - skip it entirely
- ✅ **Scripts work out of the box** - just run them
- ✅ **Eager attention is used** - standard PyTorch attention
- ⚠️ **Slightly slower** - but works reliably on Colab

## 🔍 Performance Impact

Flash attention provides ~20-30% speedup in attention operations, but:
- Installing it on Colab is difficult (requires compilation)
- Often fails with CUDA/cuDNN version mismatches
- Eager mode works reliably and is fast enough for Colab

**Bottom line**: Training will work fine without flash-attn. The time estimates already account for using eager attention.

## ✅ Verification

After running the training script, check it's using eager mode:
```python
# You should NOT see any flash_attn imports in the error logs
# You should see the model loading successfully
```

## 📚 Technical Details

### Attention Implementations Available:

1. **flash_attn** (Flash Attention 2)
   - Fastest, most memory efficient
   - Requires flash-attn package
   - Hard to install on Colab

2. **eager** (PyTorch native) ✅ **Using this**
   - Standard PyTorch attention
   - Works everywhere
   - ~20-30% slower than flash

3. **sdpa** (Scaled Dot Product Attention)
   - PyTorch 2.0+ fused attention
   - Faster than eager, slower than flash
   - Alternative option

### How the Fix Works:

The scripts now explicitly set:
```python
actor_rollout_ref.model.attn_implementation=eager
```

This tells the transformer model to use PyTorch's standard attention mechanism instead of trying to import flash_attn.

## 🆘 Still Getting the Error?

If you still see flash_attn errors after using the updated scripts:

1. **Make sure you're using the latest version**:
   ```bash
   git pull
   git checkout claude/check-qwen3-dependencies-W4BzB
   ```

2. **Verify the script has the fix**:
   ```bash
   grep "attn_implementation" run_qwen3-4b_psr_nsr_colab_free.sh
   ```
   Should output: `actor_rollout_ref.model.attn_implementation=eager`

3. **Check environment variable**:
   ```bash
   grep "VERL_DISABLE_FLASH_ATTN" run_qwen3-4b_psr_nsr_colab_free.sh
   ```
   Should output: `export VERL_DISABLE_FLASH_ATTN=1`

4. **Clear any cached modules**:
   ```python
   # Restart Colab runtime
   # Runtime → Restart runtime
   ```

---

**You're all set! No flash-attn needed for Colab training.** 🚀
