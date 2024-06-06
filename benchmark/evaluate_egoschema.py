import os
import io
import json
import argparse
import torch
import csv
from PIL import Image
import torchvision
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from tqdm import tqdm
from accelerate import Accelerator
import random
import numbers
import torch.backends.cudnn as cudnn
import sys
sys.path.append('../')
from timechat.common.logger import setup_logger
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, \
    conv_llava_llama_2
import logging
from pathlib import Path
import re


def inference_single_video(video_path, prompt, chat, args):
    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    video_list = []
    llm_message = chat.upload_video_without_audio(video_path, chat_state, video_list, n_frms=args.num_frames)

    print(f"\n\nprompt: {prompt}")
    chat.ask(prompt, chat_state)
    llm_message = chat.answer(conv=chat_state,
                              img_list=video_list,
                              num_beams=1,
                              temperature=args.temperature,
                              top_p=args.top_p,
                              max_new_tokens=512,
                              max_length=3000)[0]
    print(llm_message)
    return llm_message


def extract_answer(llm_message):
    answer = re.findall(r'[A-E]', llm_message)
    if len(answer) == 0:
        print('No answer found')
        answer = random.choice(['A', 'B', 'C', 'D', 'E'])
    else:
        answer = answer[0]
    map2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    return map2idx[answer]


def main(args):
    # load model
    device = torch.device(f"cuda:{args.gpu_id}")
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    cfg = Config(args)
    cfg.datasets_cfg.webvid.stride = args.stride
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.timechat_model_path
    model_config.window_size = args.window_size
    model_config.stride = args.stride
    args.model_type = model_config.model_type
    if args.no_lora:
        model_config.lora = False
    else:
        model_config.lora_alpha = args.lora_alpha

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()
    message = '\n' + '\n'.join([f'{k:<25}: {v}' for k, v in vars(args).items()])
    logging.info(message)

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    # Loading questions
    question_path = f"{args.anno_path}/questions.json"
    with open(question_path, 'r') as f:
        input_data = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pred_file = f"{args.output_dir}/pred.json"
    # Loading existing predictions
    if os.path.isfile(pred_file):
        with open(f"{args.output_dir}/pred.json", 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}

    if args.eval_on_subset_answers:
        subset_answers_path = f"{args.anno_path}/subset_answers.json"
        with open(subset_answers_path, 'r') as f:
            subset_answers = json.load(f)

    correct = 0
    total = 0
    for datum in tqdm(input_data):
        q_uid, _, question, option0, option1, option2, option3, option4 = datum.values()

        if args.eval_on_subset_answers and q_uid not in subset_answers:
            continue

        if q_uid not in predictions:
            video_path = os.path.join(args.video_path, f'{q_uid}.mp4')
            prompt = (f"Select the best answer to the following multiple-choice question based on the egocentric video. Respond with only the letter (A, B, C, D, or E) of the correct option. "
                      f"\nQuestion: {question}\n(A) {option0}\n(B) {option1}\n(C) {option2}\n(D) {option3}\n(E) {option4}\nThe best answer is:")
            video_llm_pred = inference_single_video(video_path, prompt, chat, args)
            answer = extract_answer(video_llm_pred)

            if args.eval_on_subset_answers:
                if answer == subset_answers[q_uid]:
                    correct += 1
                total += 1

            predictions[q_uid] = {'prompt': prompt, 'output': video_llm_pred, 'answer': answer}
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=4)

    with open(f"{args.output_dir}/output.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['q_uid', 'answer'])

        for q_uid, content in predictions.items():
            writer.writerow([q_uid, content['answer']])

    if args.eval_on_subset_answers:
        print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='../eval_configs/timechat.yaml')
    parser.add_argument('--anno_path', type=str, default='/home/v-shuhuairen/mycontainer/data/EgoSchema')
    parser.add_argument('--video_path', type=str, default='/home/v-shuhuairen/mycontainer/data/EgoSchema/good_clips_git')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--timechat_model_path',
                        default='../ckpt/timechat/train_stage2_llama2_7b_time64k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231026060/checkpoint_2.pth')
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=20)
    parser.add_argument('--eval_on_subset_answers', action='store_true')
    args = parser.parse_args()
    accelerate = Accelerator()
    main(args)
