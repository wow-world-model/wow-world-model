#!/bin/bash
BASE_MODEL_FOLDER="../dit_models/checkpoints/WoW-1-Wan-1.3B-2M"

GPU_ID=0

PORT=7860

SHARE=false

CMD="python wan_infer_demo_1.3B_i2v.py \
    --base_model_folder \"$BASE_MODEL_FOLDER\" \
    --gpu $GPU_ID \
    --port $PORT"

if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

eval $CMD
