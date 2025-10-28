#!/bin/bash
BASE_MODEL_FOLDER="../ckpt/WoW-1-Wan-1.3B-2M/"

CHECKPOINT_PATH="../ckpt/WoW-1-Wan-1.3B-2M/WoW_ckpt"

GPU_ID=0

PORT=7860

SHARE=false

CMD="python wan_infer_demo_1.3B_i2v.py \
    --base_model_folder \"$BASE_MODEL_FOLDER\" \
    --gpu $GPU_ID \
    --port $PORT \
    --checkpoint_path \"$CHECKPOINT_PATH\""

if [ "$SHARE" = true ]; then
    CMD="$CMD --share"
fi

eval $CMD
