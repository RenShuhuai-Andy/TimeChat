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
    parser.add_argument('--dataset', default='qvhighlights')  # anet
    parser.add_argument('--anno_path',
                        default='/home/yaolinli/dataset/QVhighlights/')  # /home/yaolinli/dataset/ActivityNet_asr_denseCap
    parser.add_argument('--video_path', default='QVhighlights/videos/train')  # ActivityNet_asr_denseCap/anet_6fps_224
    parser.add_argument('--outpath', default='./')
    parser.add_argument('--ratio', type=float, default=-1)
    args = parser.parse_args()
    '''output data example:
    [
        {    
            "video": "xHr8X2Wpmno.mp4", 
            "QA": [
            {"q": "You are given a video from the QVHighlights dataset. Please find the highlight moments in the video described by a sentence query, determining the highlight moments' timestamps and their saliency scores. The output format should be like: 'There are 10 highlight moments in the 82, 84, 86, 88, 90, 92, 94, 96, 98, 100 second. Their saliency scores are 1.3, 1.7, 1.7, 1.7, 1.7, 1.3, 1.7, 2.3, 2.3, 2.3'. Now I will give you the sentence query: <query_placeholder>. Please return the query-based highlight moments.",
            "a": "There are 10 highlight moments in the 82, 84, 86, 88, 90, 92, 94, 96, 98, 100 second. Their saliency scores are 1.3, 1.7, 1.7, 1.7, 1.7, 1.3, 1.7, 2.3, 2.3, 2.3." }
            ]
        },
    ]
    '''
    if args.dataset in ["qvhighlights"]:
        prompts = list(get_prompt("prompts/highlight_prompts.json").values())
    else:
        prompts = list(get_prompt("prompts/video_summarize_prompts.json").values())
    video_root = "/home/yaolinli/dataset"
    base_video_path = args.video_path
    dataset = args.dataset

    for split in ["train"]:  # "val", "test"
        if dataset == "qvhighlights":
            filename = f"{split}.caption_coco_format.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            if args.ratio > 0:
                annos = random.sample(annos, int(len(annos) * args.ratio))
            it_data = []
            for jterm in annos:
                cap = jterm["caption"]
                saliency_scores = jterm["saliency_scores"]
                relevant_clip_ids = jterm["relevant_clip_ids"]
                assert len(saliency_scores) == len(relevant_clip_ids)
                vid = os.path.join(base_video_path, jterm["image_id"].split("/")[-1])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                # format the highlight detection answer
                text_query = '\'' + cap + '\''
                mnt = len(saliency_scores)
                times = [str(round(cid * 2.0, 1)) for cid in relevant_clip_ids]
                times_str = ", ".join(times)
                scores = [str(round(sum(s) / len(s), 1)) for s in saliency_scores]
                scores_str = ", ".join(scores)
                answer = f'There are {mnt} highlight moments in the {times_str} second. Their saliency scores are {scores_str}.'
                QA = []
                prompt = random.choice(prompts)
                prompt = prompt.replace("<query_placeholder>", "{}")
                instruction = prompt.format(text_query)
                QA.append({"q": instruction.rstrip(), "a": answer.rstrip()})
                it_data.append({"video": vid, "QA": QA})

        # for dataset SumMe and TVSum, we use all the data
        if dataset in ["summe", "tvsum"]:
            filename = f"all.caption_coco_format.json"
            annos = read_json(os.path.join(args.anno_path, filename))
            if args.ratio > 0:
                annos = random.sample(annos, int(len(annos) * args.ratio))
            it_data = []
            for jterm in annos:
                saliency_scores = jterm["scores"]
                # convert score range from 0-1 to 1-5
                saliency_scores = [s * 4 + 1 for s in saliency_scores]
                timestamps = jterm["timestamps"]
                assert len(saliency_scores) == len(timestamps)
                vid = os.path.join(base_video_path, jterm["image_id"].split("/")[-1])
                # check wether the video exists
                if not os.path.exists(os.path.join(video_root, vid)):
                    continue
                # format the highlight detection answer
                mnt = len(saliency_scores)
                times = [str(round(t, 1)) for t in timestamps]
                times_str = ", ".join(times)
                scores = [str(round(s, 1)) for s in saliency_scores]
                scores_str = ", ".join(scores)
                answer = f'The highlight timestamps are in the {times_str} seconds. Their saliency scores are {scores_str}.'
                QA = []
                prompt = random.choice(prompts)
                prompt = prompt.replace("<query_placeholder>", "")
                prompt = prompt.replace("<dataset_placeholder>", "{}")
                instruction = prompt.format(dataset)
                QA.append({"q": instruction.rstrip(), "a": answer.rstrip()})
                it_data.append({"video": vid, "QA": QA})

        print(f"==> {args.dataset} dataset  \t# examples num: {len(it_data)}")
        if dataset in ["summe", "tvsum"]:
            out_name = "instruct_vhd_{}_{}.json".format(len(it_data), args.dataset)
        else:
            out_name = "instruct_vhd_{}k_{}.json".format(round(len(it_data) / 1000, 1), args.dataset)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(it_data, os.path.join(args.outpath, out_name))
