import json
import argparse
import os
from copy import deepcopy
import pdb
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


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


def write_json(data, path):
    with open(path, "w") as fout:
        json.dump(data, fout)
    return


def read_txt(path):
    data = []
    with open(path, "r") as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            # e.g. AO8RW 0.0 6.9##a person is putting a book on a shelf.
            line = line.strip("\n")
            cap = line.split("##")[-1]
            if len(cap) < 2:
                continue
            terms = line.split("##")[0].split(" ")
            vid = terms[0] + ".mp4"
            start_time = float(terms[1])
            end_time = float(terms[2])
            data.append({"image_id": vid, "caption": cap, "timestamp": [start_time, end_time], "id": i})
    return data


def process_anet(annos, video_path):
    new_annos = []
    idx = 0
    for k, v in tqdm(annos.items()):
        vid = f'{k}.mp4'
        if not os.path.exists(os.path.join(video_path, vid)):
            continue
        for timestamp, sentence in zip(v['timestamps'], v['sentences']):
            new_annos.append({"image_id": vid, "caption": sentence, "timestamp": timestamp, "id": idx})
            idx += 1
    return new_annos


def filter_sent(sent):
    sent = sent.strip(" ")
    if len(sent) < 2:
        return False
    sent = sent.replace("#", "")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='charades')  # anet
    parser.add_argument('--anno_path', default='/home/yaolinli/dataset/Charades/charades_annotation/')
    parser.add_argument('--video_path',
                        default='/home/yaolinli/dataset/Charades/videos')  # ActivityNet_asr_denseCap/anet_6fps_224
    parser.add_argument('--outpath', default='./')
    args = parser.parse_args()
    '''output data example:
    {
        "annotations": [ 
        {   
            "image_id": "3MSZA.mp4", 
            "caption": "person turn a light on.",
            "timestamp": [24.3, 30.4],
        },
        ...
        ]
    }
    '''

    for split in ["train", "val", "test"]:
        if args.dataset == "charades":
            filename = f"charades_sta_{split}.txt"
            annos = read_txt(os.path.join(args.anno_path, filename))
            data = {}
            data["annotations"] = annos
        elif args.dataset == "anet":
            mapping = {'train': 'train.json', 'val': 'val_1.json', 'test': 'val_2.json'}
            filename = mapping[split]
            annos = json.load(open(os.path.join(args.anno_path, filename), "r"))
            annos = process_anet(annos, args.video_path)
            data = {}
            data["annotations"] = annos
        else:
            print("Do not support this dataset!")
            exit(0)

        print(f"==> {args.dataset} dataset  \t# examples num: {len(annos)}")
        out_name = "{}.caption_coco_format.json".format(split)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(data, os.path.join(args.outpath, out_name))
