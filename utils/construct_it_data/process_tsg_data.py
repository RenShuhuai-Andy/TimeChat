import json
import argparse
import os
from copy import deepcopy
import pdb
import numpy as np
import random
from pathlib import Path
from decord import VideoReader
from tqdm import tqdm
import random as rnd


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


def is_video_readable(video_path):
    try:
        n_frms = 96
        vr = VideoReader(video_path)
        # vlen = len(vr)
        # start, end = 0, vlen
        # acc_samples = min(n_frms, vlen)
        # n_frms = min(n_frms, vlen)
        #
        # intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        # ranges = []
        # for idx, interv in enumerate(intervals[:-1]):
        #     ranges.append((interv, intervals[idx + 1] - 1))
        # try:
        #     indices = [rnd.choice(range(x[0], x[1])) for x in ranges]
        # except:
        #     indices = np.random.permutation(vlen)[:acc_samples]
        #     indices.sort()
        #     indices = list(indices)
        # temp_frms = vr.get_batch(indices)

        if len(vr) > 0:
            return True
        else:
            return False
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yttemporal')
    parser.add_argument('--video_path', default='yttemporal180m/videos/')
    parser.add_argument('--asr_path', default='yttemporal180m/whisper_outputs_with_time/tiny.en.cleaned/')
    parser.add_argument('--outpath', default='/home/v-shuhuairen/mycontainer/data/yttemporal180m/')
    parser.add_argument('--ratio', type=float, default=-1)
    args = parser.parse_args()
    '''output data example:
    [
        {    
            "video": "xHr8X2Wpmno.mp4", 
            "QA": [
            {"q": "Watch the video, transcribe the speech, and indicate when each segment starts and ends. Follow this format: 'start time - end time, transcribed speech'.",
            "a": "21.0 - 22.0 seconds, begin to run up.  23.0 - 24.0 seconds, begin to jump up.  25.0 - 26.0 seconds, fall to the ground. }
            ]
        },
    ]
    '''
    prompts = list(get_prompt("prompts/tsp_prompts.json").values())

    base_video_path = args.video_path
    dataset = args.dataset
    video_root = "/home/v-shuhuairen/mycontainer/data/"

    # train 1192, val 415, test 415
    for split in ["train"]:  # "val", "test"
        if dataset == "yttemporal":
            max_num_asr = 15  # only use max to 15 asr
            videos = os.listdir(os.path.join(video_root, base_video_path))
            it_data = []
            for i, video in tqdm(enumerate(videos)):
                video_path = os.path.join(video_root, base_video_path, video)
                # if not is_video_readable(video_path):
                #     print(f"video {video_path} is not readable, skip it.")
                #     continue
                asr_path = os.path.join(video_root, args.asr_path, video.split('.')[0] + '.txt')
                if not os.path.exists(asr_path):
                    print(f"asr {asr_path} does not exist, skip it.")
                    final_asr = 'None.'
                else:
                    with open(asr_path, 'r') as f:
                        asrs = f.readlines()
                    final_asr = ''
                    stride = len(asrs) // max_num_asr
                    stride = stride if stride > 0 else 1
                    for idx in range(1, len(asrs), stride):
                        asr = asrs[idx]
                        asr = asr.strip()
                        if not asr.endswith('.'):
                            asr = asr + '.'
                        asr = asr.split('\t')
                        real_timestamp_start, real_timestamp_end, caption = float(asr[0]), float(asr[1]), asr[2]
                        asr = f"{real_timestamp_start:.1f} - {real_timestamp_end:.1f} seconds, {caption} "
                        final_asr += asr
                    if final_asr == '':
                        final_asr = 'None.'
                QA = []
                instruction = random.choice(prompts)
                QA.append({"q": instruction.rstrip(), "a": f'Transcribed speech: {final_asr}'})
                vid = os.path.join(base_video_path, video)
                it_data.append({"video": vid, "QA": QA})
                if i < 5:
                    print({"video": vid, "QA": QA})
        else:
            print("Do not support this dataset!")
            exit(0)

        print(f"==> {args.dataset} dataset  \t# examples num: {len(it_data)}")
        out_name = "instruct_tsp_{}k_{}_{}asr.json".format(round(len(it_data) / 1000, 1), args.dataset, max_num_asr)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        write_json(it_data, os.path.join(args.outpath, out_name))
