#!/bin/bash

DIR="ckpt/timechat"
MODEL_DIR=${DIR}/timechat_7b.pth

WS=32
STRIDE=32

TASK='temp_compass'
ANNO_DIR='/home/v-shuhuairen/mycontainer/data/TempCompass/questions'
VIDEO_DIR='/home/v-shuhuairen/mycontainer/data/TempCompass/videos'


NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}

TASK_TYPES=('multi-choice' 'captioning' 'yes_no' 'caption_matching')

for TASK_TYPE in "${TASK_TYPES[@]}"; do
    echo "Processing task type: ${TASK_TYPE}"

    python evaluate_tempcompass.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
    --output_dir ${OUTPUT_DIR} --num_frames ${NUM_FRAME} \
    --timechat_model_path ${MODEL_DIR} \
    --window_size ${WS} --stride ${STRIDE} \
    --task_type ${TASK_TYPE} \
    --lora_alpha 20
done
