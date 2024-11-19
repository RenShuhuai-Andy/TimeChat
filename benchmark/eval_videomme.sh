#!/bin/bash

DIR="/home/wangziyue/workspace/TimeChat/ckpt/timechat"
MODEL_DIR=${DIR}/timechat_7b.pth

WS=32
STRIDE=32

TASK='videomme'
ANNO_DIR='/home/wangziyue/workspace/dataset/Video-MME'
VIDEO_DIR='/home/wangziyue/workspace/dataset/Video-MME/data'
# SUBTITLE_DIR='/home/wangziyue/workspace/dataset/Video-MME/subtitle'

NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}

python evaluate_vediomme.py  --video_path ${VIDEO_DIR} \
--output_dir ${OUTPUT_DIR} --num_frames ${NUM_FRAME} \
--timechat_model_path ${MODEL_DIR} \
--window_size ${WS} --stride ${STRIDE} \
--lora_alpha 20
