import json
import argparse
import os
import re
from copy import deepcopy
import pdb
import numpy as np
from pathlib import Path

# read json files
def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def write_json(path, data):
    with open(path, "w") as fout:
        json.dump(data, fout)
    print("The format file has been saved at:{}".format(path))
    return


def extract_time_part(time_part):
    radius = 20
    # remove 1. 2. 3. etc.
    extracted_time_part = re.compile(r"\d+\.*\d*\s*-\s*\d+\.*\d*").findall(time_part)
    if len(extracted_time_part) == 0:
        if time_part.count(':') == 1:
            # for 1. The video starts at 0:00.
            extracted_time = re.compile(r"\d+\.*\d*:\d+\.*\d*").findall(time_part)[0]
            extracted_time = int(extracted_time.split(':')[0]) * 60 + int(extracted_time.split(':')[1])
            if extracted_time > radius:
                extracted_time_part = [f'{extracted_time - radius} - {extracted_time + radius}']
            else:
                extracted_time_part = [f'{extracted_time} - {extracted_time + 2*radius}']
        elif time_part.count(':') == 2:
            # for * Using a wok to cook dishes (from 1:09 to 1:20)
            start, end = re.compile(r"\d+\.*\d*:\d+\.*\d*").findall(time_part)
            start_seconds = int(start.split(':')[0]) * 60 + int(start.split(':')[1])
            end_seconds = int(end.split(':')[0]) * 60 + int(end.split(':')[1])
            extracted_time_part = [f'{start_seconds} - {end_seconds}']
        else:
            pass
    if len(extracted_time_part) == 0:
        extracted_time_part = re.compile(r"\d+\.*\d*(?!\.)").findall(time_part)
        if len(extracted_time_part) == 1:
            # for start - 180 seconds, Add 1/4 cup of olive oil to the pan
            extracted_time = float(extracted_time_part[0])
            if extracted_time > radius:
                extracted_time_part = [f'{extracted_time - radius} - {extracted_time + radius}']
            else:
                extracted_time_part = [f'{extracted_time} - {extracted_time + 2 * radius}']
        elif len(extracted_time_part) == 2:
            # for 10s-38s, Sharpener was instructed to get a pair of scissors and cut off the top of the pineapple.
            extracted_time_part = [f'{extracted_time_part[0]} - {extracted_time_part[1]}']
        else:
            pass
    return extracted_time_part


def extract_time_from_para(paragraph):
    paragraph = paragraph.lower()
    patterns = [
        (r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)", r"(\d+\.*\d*\s*-\s*\d+\.*\d*)")  # n - m, caption
    ]
    timestamps = []
    captions = []

    # Check for m - n, captions (no seconds)
    for time_pattern, string_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph)
        string_matches = re.findall(string_pattern, paragraph)

        if time_matches:
            # n - m, caption
            timestamps = [[float(start), float(end)] for start, end in time_matches]
            # get captions
            rest_para = paragraph
            for time_string in string_matches:
                rest_para = rest_para.replace(time_string, '\n')
            captions = rest_para.replace('seconds', '').split('\n')
        if len(timestamps) > 0:
            break
            
    # Check for 'Start time: N seconds' and 'End time: M seconds' format, e.g.
    #   4. Start time: 113 seconds
    #   End time: 116 seconds
    #   Description: Spreading brownies in a pan
    if len(timestamps) == 0:
        start_time_pattern = r"(?:start(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)"
        end_time_pattern = r"(?:end(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)"
        end_matches = re.findall(end_time_pattern, paragraph, re.DOTALL | re.IGNORECASE)
        start_matches = re.findall(start_time_pattern, paragraph, re.DOTALL | re.IGNORECASE)
    
        if start_matches and end_matches:
            timestamps = [[float(start), float(end)] for start, end in zip(start_matches, end_matches)]
            captions = re.findall(r"description: (.*)", paragraph)
            if len(captions) == 0:
                captions = re.findall(r"\*\s*(.*)", paragraph)
            
    # Check for 'start time X.X, end time Y.Y' format
    if len(timestamps) == 0:
        start_end_matches = re.findall(r"start time (\d+\.*\d*), end time (\d+\.*\d*)", paragraph)
        if start_end_matches:
            timestamps = list(start_end_matches)
            for (start, end) in start_end_matches:
                paragraph = paragraph.replace(f'start time {start}, end time {end}', '\n')
                captions = paragraph.split('\n')
            if len(timestamps) > 0:
                pdb.set_trace()
    
    captions = [c.strip().strip(", ").rstrip() for c in captions if len(c) > 5]
    min_len = min(len(timestamps), len(captions))
    timestamps = timestamps[:min_len]
    captions = captions[:min_len]
    assert len(timestamps) == len(captions), f"# timestamps {len(timestamps)}, # captions {len(captions)}, para {paragraph}."
    return timestamps, captions


def format_dvc_output(caption):
    timestamps = []
    sents = []
    # type 1: directly detect timestamps in generated paragraph to process multi-lines cases like:
    #   1. Start time: 105 
    #   End time: 109 
    #   Description: Making brown sugar sandwiches with white bread
    paras = caption
    timestamps, sents = extract_time_from_para(paras)
        
    # type 2：detect timestamps in splited sentences
    if len(timestamps) == 0:
        if '\n' in caption:
            caps = caption.split('\n')
            caps = [c for c in caps if len(c) > 7]
        else:
            raw_caps = caption.split('.')
            caps = [c for c in raw_caps if len(c) > 7]
            caps = [c+'.' for c in caps]
        for cap in caps:
            try:            
                if len(timestamps) == 0:
                    parts = cap.split('seconds')
                    parts = [p.strip(',') for p in parts]
                    time_part = parts[0]
                    extracted_time_part = extract_time_part(time_part)
                    if len(extracted_time_part) == 0:
                        continue
                    else:
                        time_part = extracted_time_part[0]
                    sent_part = parts[-1]
                    stime = round(float(time_part.split('-')[0].strip()), 2)
                    etime = round(float(time_part.split('-')[1].strip()), 2)
                    timestamps.append([stime, etime])
                    sents.append(sent_part.strip())
            except:
                continue

    return timestamps, sents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default='/home/yaolinli/code/Ask-Anything/video_chat/output/eval_7b_instruct1.2k_youcook2_bz8_f8_epoch3/val_f8_result.json')
    parser.add_argument('--outpath', default='results')
    args = parser.parse_args()
    
    datas = read_json(args.inpath)
    # example in output file
    # {
    #     "xHr8X2Wpmno.mp4": [
    #         {
    #             "timestamp": [47.0, 60.0],
    #             "caption": "a person is shown tying a plant into a bun and putting the bun into a pink jar."
    #         }, 
    #         ...
    #     ]
    #     ...
    # }
    fmt_datas = {}
    count = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        # if vid != 'eHk6NSLGAkc.mp4':
        #     continue
        caption = jterm["generated_cap"]
        timestamps, sents = format_dvc_output(caption)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, caption)
        fmt_datas[vid] = []
        for j in range(len(timestamps)):
            fmt_datas[vid].append({"timestamp": timestamps[j], "caption": sents[j]})
        count.append(len(timestamps))
    
    print(f"predict avg {sum(count)/len(count)} events per video")
    print(f'parse failed number: {cnt}')
    split = args.inpath.split('/')[-1].split('_')[0]
    out_file = args.inpath.split('/')[-2]
    out_path = f'{out_file}_{split}.json'
    if args.outpath:
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(args.outpath, out_path)
    write_json(os.path.join(os.getcwd(), out_path), fmt_datas)