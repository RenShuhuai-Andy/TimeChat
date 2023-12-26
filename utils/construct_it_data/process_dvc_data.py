import json
import argparse
import os
from copy import deepcopy
import pdb
import numpy as np
import random
from pathlib import Path


def read_txt(path):
    data = []
    with open(path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(line.strip("\n"))
    print("data samples num: ", len(data))
    return data


# read json files
def read_json(path, key=True):
    with open(path, "r", encoding="utf-8") as fin:
        datas = json.load(fin)
        if key:
            annos = datas["annotations"]
        else:
            annos = datas
    return annos


def get_prompt(path):
    with open(path, "r", encoding="utf-8") as fin:
        datas = json.load(fin)
    return datas


def write_json(data, path):
    with open(path, "w") as fout:
        json.dump(data, fout)
    return


def get_max_time(segments):
    all_times = []
    for timestamp in segments:
        all_times.extend(timestamp)
    return max(all_times)


def filter_sent(sent):
    if len(sent) < 2:
        return False
    sent = sent.replace("#", "")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='youcook2')  # anet
    parser.add_argument('--anno_path',
                        default='/home/yaolinli/dataset/YouCook2_asr_denseCap/')  # /home/yaolinli/dataset/ActivityNet_asr_denseCap
    parser.add_argument('--video_path',
                        default='YouCook2_asr_denseCap/youcook2_6fps_224')  # ActivityNet_asr_denseCap/anet_6fps_224
    parser.add_argument('--outpath', default='./')
    parser.add_argument('--ratio', type=float, default=-1)
    args = parser.parse_args()
    '''output data example:
    [
        {    
            "video": "xHr8X2Wpmno.mp4", 
            "QA": [
            {"q": "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with a sentence.",
            "a": "21.0 - 22.0 seconds, begin to run up.  23.0 - 24.0 seconds, begin to jump up.  25.0 - 26.0 seconds, fall to the ground. }
            ]
        },
    ]
    '''
    prompts = list(get_prompt("prompts/dvc_prompts.json").values())

    base_video_path = args.video_path
    dataset = args.dataset
    video_root = "/home/yaolinli/dataset"

    # train 1192, val 415, test 415
    for split in ["train"]:  # "val", "test"
        if dataset == "youcook2":
            filename = f"{split}.caption_coco_format.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            it_data = []
            for jterm in annos:
                vid = os.path.join(base_video_path, jterm["image_id"])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                cap = jterm["pure_cap"]
                segments = jterm["segments"]  # a list
                sents = cap.split(".")
                sent_with_tsmp = []
                for i, sent in enumerate(sents):
                    sent = sent.strip(" ")
                    new_sent = f'{round(float(segments[i][0]), 1)} - {round(float(segments[i][1]), 1)} seconds, {sent}. '
                    sent_with_tsmp.append(new_sent)
                cap_with_tsmp = " ".join(sent_with_tsmp)
                QA = []
                instruction = random.choice(prompts)
                QA.append({"q": instruction.rstrip(), "a": cap_with_tsmp.rstrip()})
                it_data.append({"video": vid, "QA": QA})
        elif dataset == "anet":
            filename = f"{split}.caption_coco_format.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            if args.ratio > 0:
                data_num = int(len(annos) * args.ratio)
                annos = annos[:data_num]
            it_data = []
            for jterm in annos:
                vid = os.path.join(base_video_path, jterm["image_id"].split("/")[-1])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                cap = jterm["caption"]
                segments = jterm["segments"]  # a list
                sents = cap.split(".")
                sent_with_tsmp = []
                for i, sent in enumerate(sents):
                    sent = sent.strip(" ")
                    if len(sent) < 1:
                        continue
                    new_sent = f'{round(float(segments[i][0]), 1)} - {round(float(segments[i][1]), 1)} seconds, {sent}. '
                    sent_with_tsmp.append(new_sent)
                cap_with_tsmp = " ".join(sent_with_tsmp)

                QA = []
                instruction = random.choice(prompts)
                QA.append({"q": instruction.rstrip(), "a": cap_with_tsmp.rstrip()})
                it_data.append({"video": vid, "QA": QA})
        elif dataset == "vitt":
            # this dataset only has start time in timestamp annotations
            # e.g. [8.46, "Ingredients needed"]
            prompts = list(get_prompt("prompts/dvc_prompts_start-time-only.json").values())
            filename = f"{split}.json"
            annos = read_json(os.path.join(args.anno_path, filename), key=False)
            if args.ratio > 0:
                data_num = int(len(annos) * args.ratio)
                annos = annos[:data_num]
            it_data = []
            for jterm in annos:
                vid = os.path.join(base_video_path, jterm["vid"])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                sents = jterm["captions"]
                segments = jterm["segments"]  # a list
                sent_with_tsmp = []
                for i, sent in enumerate(sents):
                    sent = filter_sent(sent)
                    if not sent:
                        continue
                    new_sent = f'{round(float(segments[i][0]), 1)} seconds, {sent}. '
                    sent_with_tsmp.append(new_sent)
                cap_with_tsmp = " ".join(sent_with_tsmp)
                QA = []
                instruction = random.choice(prompts)
                QA.append({"q": instruction.rstrip(), "a": cap_with_tsmp.rstrip()})
                it_data.append({"video": vid, "QA": QA})
        elif dataset == "queryd":
            split_file = f"exists_{split}_list.txt"
            split_vids = read_txt(os.path.join(args.anno_path, split_file))
            cap_file = "raw_captions_combined_filtered.json"
            time_file = "times_captions_combined_filtered.json"
            anno_caps = read_json(os.path.join(args.anno_path, cap_file), key=False)
            anno_times = read_json(os.path.join(args.anno_path, time_file), key=False)
            assert len(anno_times) == len(anno_caps)
            if args.ratio > 0:
                print("Error! This dataset is not applicable")
                exit(0)
            it_data = []
            for vname in anno_times.keys():
                if vname not in split_vids:
                    continue
                vid = os.path.join(base_video_path, vname)
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                caps = anno_caps[vname]
                segments = anno_times[vname]  # a list
                # filter long videos > 300 seconds
                max_time = get_max_time(segments)
                if max_time > 300:
                    continue
                sents = [" ".join(c) for c in caps]
                assert len(sents) == len(segments)
                sent_with_tsmp = []
                for i, sent in enumerate(sents):
                    sent = sent.strip(" ")
                    if len(sent) < 1:
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
        out_name = "instruct_dvc_{}k_{}.json".format(round(len(it_data) / 1000, 1), args.dataset)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(it_data, os.path.join(args.outpath, out_name))
