#!/bin/bash

DIR="ckpt/timechat"
MODEL_DIR=${DIR}/timechat_7b.pth

WS=32
STRIDE=32

TASK='mvbench'
ANNO_DIR='/home/v-shuhuairen/mycontainer/data/MVBench/json'
VIDEO_DIR='/home/v-shuhuairen/mycontainer/data/MVBench/video'


NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}

python evaluate_mvbench.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
--output_dir ${OUTPUT_DIR} --num_frames ${NUM_FRAME} \
--timechat_model_path ${MODEL_DIR} \
--window_size ${WS} --stride ${STRIDE}
