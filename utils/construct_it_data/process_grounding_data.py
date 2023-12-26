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
        annos = datas["annotations"]
    return annos


def read_jsonl(path):
    anno = []
    with open(path, "r") as fin:
        datas = fin.readlines()
        for data in datas:
            anno.append(json.loads(data.strip()))
    return anno


def get_prompt(path):
    with open(path, "r", encoding="utf-8") as fin:
        datas = json.load(fin)
    return datas


def write_json(data, path):
    with open(path, "w") as fout:
        json.dump(data, fout)
    return


def read_txt(path):
    data = []
    with open(path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            # e.g. AO8RW 0.0 6.9##a person is putting a book on a shelf.
            line = line.strip("\n")
            cap = line.split("##")[-1]
            if len(cap) < 2:
                continue
            terms = line.split("##")[0].split(" ")
            vid = terms[0] + ".mp4"
            start_time = terms[1]
            end_time = terms[2]
            data.append({"image_id": vid, "caption": cap, "timestamp": [start_time, end_time]})
    return data


def filter_sent(sent):
    sent = sent.strip(" ")
    if len(sent) < 2:
        return False
    sent = sent.replace("#", "")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='anet')  # anet
    parser.add_argument('--anno_path',
                        default='/home/yaolinli/dataset/ActivityNet_asr_denseCap/')  # /home/yaolinli/dataset/ActivityNet_asr_denseCap
    parser.add_argument('--video_path',
                        default='ActivityNet_asr_denseCap/anet_6fps_224')  # ActivityNet_asr_denseCap/anet_6fps_224
    parser.add_argument('--outpath', default='./')
    parser.add_argument('--ratio', type=float, default=-1)
    args = parser.parse_args()
    '''output data example:
    [
        {    
            "video": "xHr8X2Wpmno.mp4", 
            "QA": [
            {"q": "Localize the visual content described by the given textual query <query_placeholder> in the video, and output the start and end timestamps in seconds.",
            "a": "The given query happens in the 12 seconds - 16 seconds. }
            ]
        },
    ]
    '''
    prompts = list(get_prompt("prompts/video_grounding_prompts.json").values())

    video_root = "/home/yaolinli/dataset"
    base_video_path = args.video_path
    dataset = args.dataset

    for split in ["train"]:  # "val", "test"
        if dataset == "anet":
            filename = f"{split}.caption_coco_format.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            if args.ratio > 0:
                data_num = int(len(annos) * 0.5)
                annos = annos[data_num:]
                annos = random.sample(annos, int(len(annos) * args.ratio))
            it_data = []
            for jterm in annos:
                cap = jterm["caption"]
                segments = jterm["segments"]  # a list
                sents = cap.split(".")
                vid = os.path.join(base_video_path, jterm["image_id"].split("/")[-1])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                for i, sent in enumerate(sents):
                    sent = filter_sent(sent)
                    if not sent:
                        continue
                    text_query = '\'' + sent + '\''
                    answer = f'The given query happens in {round(float(segments[i][0]), 1)} - {round(float(segments[i][1]), 1)} seconds.'
                    QA = []
                    prompt = random.choice(prompts)
                    prompt = prompt.replace("<query_placeholder>", "{}")
                    instruction = prompt.format(text_query)
                    QA.append({"q": instruction.rstrip(), "a": answer.rstrip()})
                    it_data.append({"video": vid, "QA": QA})
        elif dataset == "charades":
            filename = f"charades_sta_{split}.txt"
            annos = read_txt(os.path.join(args.anno_path, filename))
            if args.ratio > 0:
                annos = random.sample(annos, int(len(annos) * args.ratio))
            it_data = []
            for jterm in annos:
                cap = jterm["caption"]
                cap = filter_sent(cap)
                if not cap:
                    continue
                timestamp = jterm["timestamp"]  # a list
                vid = os.path.join(base_video_path, jterm["image_id"])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    print(f"video {vid} do not exist")
                    continue
                text_query = '\'' + cap.strip(".") + '\''
                answer = f'The given query happens in {timestamp[0]} - {timestamp[1]} seconds.'
                QA = []
                prompt = random.choice(prompts)
                prompt = prompt.replace("<query_placeholder>", "{}")
                instruction = prompt.format(text_query)
                QA.append({"q": instruction.rstrip(), "a": answer.rstrip()})
                it_data.append({"video": vid, "QA": QA})
        elif dataset == "didemo":
            gap = 5  # this dataset split each video to 5-second clips
            filename = f"{split}.jsonl"
            annos = read_jsonl(os.path.join(args.anno_path, filename))
            if args.ratio > 0:
                annos = random.sample(annos, int(len(annos) * args.ratio))
            it_data = []
            for jterm in annos:
                sents = jterm["caption_list"]
                segments = jterm["timestamps"]  # a list
                vid = os.path.join(base_video_path, jterm["clip_name"] + ".mp4")
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                for i, sent in enumerate(sents):
                    sent = filter_sent(sent)
                    if not sent:
                        continue
                    text_query = '\'' + sent + '\''
                    answer = f'The given query happens in {segments[i][0] * gap} - {(segments[i][1] + 1) * gap} seconds.'  # i*5 - (j+1) * 5
                    QA = []
                    prompt = random.choice(prompts)
                    prompt = prompt.replace("<query_placeholder>", "{}")
                    instruction = prompt.format(text_query)
                    QA.append({"q": instruction.rstrip(), "a": answer.rstrip()})
                    it_data.append({"video": vid, "QA": QA})
        else:
            print("Do not support this dataset!")
            exit(0)

        print(f"==> {args.dataset} dataset  \t# examples num: {len(it_data)}")
        out_name = "instruct_tvg_{}k_{}.json".format(round(len(it_data) / 1000, 1), args.dataset)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(it_data, os.path.join(args.outpath, out_name))
