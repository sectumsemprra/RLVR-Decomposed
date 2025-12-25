#!/bin/bash
# Wrapper script to show clear progress during training
# Use this for better visibility of training status

echo "=================================================="
echo "🚀 RLVR-Decomposed Training Starting"
echo "=================================================="
echo ""
echo "📊 Configuration:"
echo "  - Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "  - Model: Will be shown during startup"
echo "  - Total Epochs: 20"
echo "  - Checkpoints saved every: 7 epochs"
echo ""
echo "=================================================="
echo "⏰ Training will start in 3 seconds..."
echo "=================================================="
sleep 3

# Run the actual training script with unbuffered output
# and pipe to both screen and log file
stdbuf -oL -eL bash "$@" 2>&1 | tee -a training_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Check ./checkpoints/ for saved models"
else
    echo "❌ Training exited with error code: $EXIT_CODE"
    echo "📋 Check the log file above for details"
fi
echo "=================================================="

exit $EXIT_CODE
