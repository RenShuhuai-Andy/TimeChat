#!/bin/bash

DIR="ckpt/timechat/train_stage2_llama2_7b_time73k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231107115"
MODEL_DIR=${DIR}/checkpoint_2.pth

TASK='dvc'
ANNO_DIR='data/TimeIT/data/dense_video_captioning/youcook2'
VIDEO_DIR='data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/'
DATASET='youcook'
SPLIT='val'
PROMPT_FILE="prompts/${TASK}_description_zeroshot.txt"
GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
ASR_DIR='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/'

#TASK='tvg'
#ANNO_DIR='data/TimeIT/data/temporal_video_grounding/charades/charades_annotation'
#VIDEO_DIR='data/Charades/videos/'
#DATASET='charades'
#SPLIT='test'
#PROMPT_FILE="prompts/${TASK}_description_zeroshot.txt"
#GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"
#ASR_DIR='data/Charades/whisper_outputs_with_time/tiny.en.cleaned/'

#TASK='vhd'
#ANNO_DIR='data/TimeIT/data/video_highlight_detection/qvhighlights/annotations_raw'
#VIDEO_DIR='data/QVhighlights/videos/val/'
#DATASET='qvhighlights'
#SPLIT='val'
#PROMPT_FILE="prompts/${TASK}_description.txt"
#GT_FILE="${ANNO_DIR}/highlight_${SPLIT}_release.jsonl"
#ASR_DIR='data/QVhighlights/whisper_outputs_with_time/tiny.en.cleaned/val/'

NUM_FRAME=96
OUTPUT_DIR=${DIR}/${TASK}

python evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} --num_frames ${NUM_FRAME} --batch_size 16 \
--prompt_file ${PROMPT_FILE} --timechat_model_path ${MODEL_DIR} \
#--asr --asr_path ${ASR_DIR}
#--debug

cd metrics/${TASK}
python eval_${TASK}.py --pred_file "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json" --gt_file ${GT_FILE} | tee "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.txt"
cd ../..