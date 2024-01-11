import json
import argparse
import os
from copy import deepcopy
import pdb
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from decord import VideoReader
import decord


def pass_video_check(video_path):
    # check if the video exists
    if not os.path.exists(video_path):
        print(f"==> {video_path} does not exist!")
        return False
    # check if the video is broken
    try:
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=224, width=224)
        vlen = len(vr)
        indices = np.arange(0, vlen).astype(int).tolist()
        temp_frms = vr.get_batch(indices)
        return True
    except:
        print(f"==> {video_path} is broken!")
        return False


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


def process_charades_tvg(anno_path, video_path):
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
    data = []
    with open(anno_path, "r") as fin:
        lines = fin.readlines()
        for i, line in enumerate(lines):
            # e.g. AO8RW 0.0 6.9##a person is putting a book on a shelf.
            line = line.strip("\n")
            cap = line.split("##")[-1]
            if len(cap) < 2:
                continue
            terms = line.split("##")[0].split(" ")
            vid = terms[0] + ".mp4"
            if not pass_video_check(os.path.join(video_path, vid)):
                continue
            start_time = float(terms[1])
            end_time = float(terms[2])
            data.append({"image_id": vid, "caption": cap, "timestamp": [start_time, end_time], "id": i})
    return data


def process_anet_dvc(annos, video_path):
    '''output data example:
        {
            "annotations": [
            {
                "image_id": "3MSZA.mp4",
                "duration": 206.86,
                "segments": [[47, 60], [67, 89], [91, 98], [99, 137], [153, 162], [163, 185]],
                "caption": "pick the ends off the verdalago. ...
            },
            ...
            ]
        }
    '''
    new_annos = []
    idx = 0
    for k, v in tqdm(annos.items()):
        vid = f'{k}.mp4'
        if not pass_video_check(os.path.join(video_path, vid)):
            continue
        new_annos.append({"image_id": vid, "caption": ''.join(v['sentences']), "segments": v['timestamps'],
                          "duration": v['duration'], "id": idx})
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
    parser.add_argument('--dataset', default='charades', choices=['charades', 'anet'], help='dataset name')
    parser.add_argument('--anno_path', help='annotation path')
    parser.add_argument('--video_path', help='video path')
    parser.add_argument('--outpath', default='./', help='output path')
    args = parser.parse_args()

    for split in ["test", "val", "train"]:
        if args.dataset == "charades":
            filename = f"charades_sta_{split}.txt"
            annos = process_charades_tvg(os.path.join(args.anno_path, filename), args.video_path)
            data = {}
            data["annotations"] = annos
        elif args.dataset == "anet":
            mapping = {'train': 'train.json', 'val': 'val_1.json', 'test': 'val_2.json'}
            filename = mapping[split]
            annos = json.load(open(os.path.join(args.anno_path, filename), "r"))
            annos = process_anet_dvc(annos, args.video_path)
            data = {}
            data["annotations"] = annos
        else:
            print("Do not support this dataset!")
            exit(0)

        print(f"==> {args.dataset} dataset  \t# examples num: {len(annos)}")
        out_name = "{}.caption_coco_format.json".format(split)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(data, os.path.join(args.outpath, out_name))
