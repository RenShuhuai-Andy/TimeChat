import os
import io
import json
import argparse
import torch
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


def inference_single_video(video_path, prompt, chat, args):
    chat_state = conv_llava_llama_2.copy()
    chat_state.system = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    video_list = []
    llm_message = chat.upload_video_without_audio(video_path, chat_state, video_list, n_frms=args.num_frames)

    prompt = f"Question: {prompt}"
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


def main(args):
    answer_prompt = {
        # "multi-choice": "\nBest Option:",     # The old version
        "multi-choice": "\nPlease directly give the best option:",
        "yes_no": "\nPlease answer yes or no:",
        # "caption_matching": "\nBest Option:",     #The old version
        "caption_matching": "\nPlease directly give the best option:",
        "captioning": ""  # The answer "Generated Caption:" is already contained in the question
    }

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
    question_path = f"{args.anno_path}/{args.task_type}.json"
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pred_file = f"{args.output_dir}/{args.task_type}.json"
    # Loading existing predictions
    if os.path.isfile(pred_file):
        with open(f"{args.output_dir}/{args.task_type}.json", 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}

    for vid, data in tqdm(input_datas.items()):
        if vid not in predictions:
            predictions[vid] = {}
            video_path = os.path.join(args.video_path, f'{vid}.mp4')
            for dim, questions in data.items():
                predictions[vid][dim] = []
                for question in questions:
                    if args.task_type == 'caption_matching':
                        # question example: "Which description is a more suitable match for the video?\nOption 1: The man is dribbling a basketball.\nOption 2: A man is dunking a basketball."
                        options = question['question'].split('\n')[1:]
                        options = [o.split(':')[0] for o in options]
                        inp = question['question'] + answer_prompt[args.task_type].replace(':', f" ({' or '.join(options)}):")
                    else:
                        inp = question['question'] + answer_prompt[args.task_type]
                    video_llm_pred = inference_single_video(video_path, inp, chat, args)
                    # while not any(prefix in video_llm_pred for prefix in ['A', 'B', 'C', 'D']):
                    #     video_llm_pred = inference_single_video(video_path, inp, chat, args)
                    predictions[vid][dim].append(
                        {'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='../eval_configs/timechat.yaml')
    parser.add_argument('--anno_path', type=str, default='/home/v-shuhuairen/mycontainer/data/TempCompass/questions')
    parser.add_argument('--video_path', type=str, default='/home/v-shuhuairen/mycontainer/data/TempCompass/videos')
    parser.add_argument('--task_type', default='multi-choice',
                        choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--timechat_model_path',
                        default='../ckpt/timechat/train_stage2_llama2_7b_time64k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231026060/checkpoint_2.pth')
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=20)
    args = parser.parse_args()
    accelerate = Accelerator()
    main(args)
