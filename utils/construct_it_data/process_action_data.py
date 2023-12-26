import json
import argparse
import os
from copy import deepcopy
import pdb
import numpy as np
import random
from pathlib import Path


# read json files
def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def get_prompt(path):
    with open(path, "r", encoding="utf-8") as fin:
        datas = json.load(fin)
    return datas


def write_json(data, path):
    with open(path, "w") as fout:
        json.dump(data, fout)
    return


def filter_sent(sent):
    if len(sent) < 2:
        return False
    sent = sent.replace("#", "")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coin')  # anet
    parser.add_argument('--anno_path',
                        default='/home/yaolinli/dataset/COIN/annotations/')  # /home/yaolinli/dataset/ActivityNet_asr_denseCap
    parser.add_argument('--video_path', default='COIN/videos_ali')  # ActivityNet_asr_denseCap/anet_6fps_224
    parser.add_argument('--outpath', default='./')
    parser.add_argument('--ratio', type=float, default=-1)
    args = parser.parse_args()
    '''output data example:
    [
        {    
            "video": "xHr8X2Wpmno.mp4", 
            "QA": [
            {"q": "Localize a series of action steps in the given video, output a start and end timestamp for each step, and briefly describe the step. ",
            "a": "t1 - t2 seconds, action 1 ... }
            ]
        },
    ]
    '''
    prompts = list(get_prompt("prompts/action_locate_prompts.json").values())

    base_video_path = args.video_path
    dataset = args.dataset
    video_root = "/home/yaolinli/dataset"

    # train 1192, val 415, test 415
    for split in ["train"]:  # "val", "test"
        if dataset == "coin":
            filename = f"{split}.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            it_data = []
            for jterm in annos:
                vid = os.path.join(base_video_path, jterm["video_path"])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    print("video {} not exists!".format(jterm[vid]))
                    continue
                segments = jterm["segments"]  # a list
                labels = jterm["labels"]
                sent_with_tsmp = []
                for i, sent in enumerate(labels):
                    sent = filter_sent(sent)
                    if not sent:
                        continue
                    new_sent = f'{round(float(segments[i][0]), 1)} - {round(float(segments[i][1]), 1)} seconds, {sent}. '
                    sent_with_tsmp.append(new_sent)
                cap_with_tsmp = " ".join(sent_with_tsmp)
                QA = []
                instruction = random.choice(prompts)
                QA.append({"q": instruction.rstrip(), "a": cap_with_tsmp.rstrip()})
                it_data.append({"video": vid, "QA": QA})
        elif dataset == "hirest":
            filename = f"{split}.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            it_data = []
            for jterm in annos:
                vid = os.path.join(base_video_path, jterm["image_id"])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    print("video {} not exists!".format(os.path.join(video_root, vid)))
                    continue
                segments = jterm["segments"]  # a list
                labels = jterm["labels"]
                sent_with_tsmp = []
                for i, sent in enumerate(labels):
                    sent = filter_sent(sent)
                    if not sent:
                        continue
                    new_sent = f'{round(float(segments[i][0]), 1)} - {round(float(segments[i][1]), 1)} seconds, {sent}. '
                    sent_with_tsmp.append(new_sent)
                cap_with_tsmp = " ".join(sent_with_tsmp)
                QA = []
                instruction = random.choice(prompts)
                QA.append({"q": instruction.rstrip(), "a": cap_with_tsmp.rstrip()})
                it_data.append({"video": vid, "QA": QA})
        else:
            print("Do not support this dataset!")
            exit(0)

        print(f"==> {args.dataset} dataset  \t# examples num: {len(it_data)}")
        out_name = "instruct_action_{}k_{}.json".format(round(len(it_data) / 1000, 1), args.dataset)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(it_data, os.path.join(args.outpath, out_name))
