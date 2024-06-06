#!/bin/bash

DIR="ckpt/timechat"
MODEL_DIR=${DIR}/timechat_7b.pth

WS=32
STRIDE=32

TASK='ego_schema'
ANNO_DIR='/home/v-shuhuairen/mycontainer/data/EgoSchema'
VIDEO_DIR='/home/v-shuhuairen/mycontainer/data/EgoSchema/good_clips_git'


NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}


python evaluate_egoschema.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
--output_dir ${OUTPUT_DIR} --num_frames ${NUM_FRAME} \
--timechat_model_path ${MODEL_DIR} \
--window_size ${WS} --stride ${STRIDE} \
--lora_alpha 20 \
--eval_on_subset_answers
