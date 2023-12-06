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


def parse_highlights(output):
    """
    Parses the model's output to extract highlight moments and their saliency scores.

    Parameters:
    output (str): The string output from the model.

    Returns:
    list: A list of dictionaries, each containing a timestamp and a saliency score.
    """
    output = output.replace('th', ' ')
    highlights = []
    timestamps = []
    scores = []
    second_num = output.count('second')
    score_num = output.count('score')
    if second_num == 0 or score_num == 0:
        return highlights, timestamps, scores
    
    # Class 1:  Handle the format where timestamps and saliency scores are listed separately.
    if second_num < 3 and score_num < 3:
        # Extract numbers and split them into timestamps and saliency scores.
        paras = output.split('\n')
        paras = [cap for cap in paras if ('second' in cap or 'score' in cap)]
        for para in paras:
            # Extract and parse the number of highlights.
            num = re.search(r"There are (\d+) highlight moments", para)
            if num:
                num_highlights = int(num.group(1))
                para = para.replace(f"There are {num_highlights} highlight moments", "")
            else:
                num_highlights = 100
            timestamp_part = para.split("score")[0]
            score_part = para.split("score")[1] if len(para.split("score")) > 1 else para
            time_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", timestamp_part)
            score_numbers = re.findall(r"\b\d\.\d\b", score_part)
            # Correctly split the numbers into timestamps and scores based on the number of highlights.
            if num_highlights:
                timestamps.extend([float(num) for num in time_numbers[:min(num_highlights, len(time_numbers))]])
                scores.extend([float(num) for num in score_numbers[:min(num_highlights, len(score_numbers))]])              


    # Class 2: Handle the detailed description format where each highlight is described one by one.
    else:
        # Split the output into individual highlight descriptions.
        descriptions = re.split(r'\n+', output)
        cands = []
        for desc in descriptions:
            if 'second' not in desc and 'score' not in desc and ':' not in desc:
                continue
            else:
                cands.append(desc)
        # search saliency score x.x
        for output in cands:
            saliency_score = re.findall(r'\b\d\.\d\b', output)
            if len(saliency_score) > 0:
                score = saliency_score[-1]
                score = score[-1] if isinstance(score, tuple) else score
                scores.append(float(score))
        
            times = re.findall(r'\b((\d{1,2}:\d{2}:\d{2}))\b|\b(\d+\.\d+\b|\b\d+)\b', output)
            filter_time = []
            for timestamp in times:
                # deal with time type like 00:00:20
                if ':' in timestamp:
                    parts = list(map(int, timestamp.split(':')))
                    if len(parts) == 3:
                        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]  # for "hh:mm:ss" format
                    else:
                        seconds = parts[0] * 60 + parts[1]  # for "mm:ss" format
                    filter_time.append(seconds)
                else:
                    timestamp = timestamp[-1] if isinstance(timestamp, tuple) else timestamp
                    if timestamp != '' and float(timestamp) > 10.0:
                        filter_time.append(timestamp)
            if len(filter_time) > 0:
                timestamps.append(filter_time[-1])

    timestamps = [float(t) for t in timestamps]
    scores = [float(s) for s in scores]
    time_len = len(timestamps)
    if time_len > 1:
        score_len = len(scores)
        if score_len < time_len:
            scores.extend([0.0] * (time_len - score_len))
        else:
            scores = scores[:time_len]
        # Combine the timestamps and scores into the result.
        highlights = [{"timestamp": ts, "saliency_score": sc} for ts, sc in zip(timestamps, scores)]
    return highlights, timestamps, scores


def format_vhd_output(paras, gts):
    highlights, timestamps, scores = parse_highlights(paras)
    # map timestamps and scores to clip ids
    gt_duration = gts["duration"]
    clip_num = int(gt_duration/2)
    clip_scores = []
    cid2score = np.zeros(clip_num)
    cid2num = np.zeros(clip_num)
    for (t, s) in zip(timestamps, scores):
        if t > gt_duration:
            continue
        clip_id = max(0, int((t-1)/2))
        cid2score[clip_id] += s
        cid2num[clip_id] += 1
    for cid in range(clip_num):
        if cid2num[cid] == 0:
            clip_scores.append(0.0)
        else:
            clip_scores.append(cid2score[cid]/cid2num[cid])
    return highlights, clip_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default='/home/yaolinli/code/Ask-Anything/video_chat/output/eval_stage2_7b_qvhighlights_bz8_f8_epoch3/qvhighlights_val_f8_result.json')
    parser.add_argument('--gtpath', default='/home/yaolinli/dataset/QVhighlights/val.caption_coco_format.json')
    parser.add_argument('--outpath', default='')
    args = parser.parse_args()
    
    datas = read_json(args.inpath)
    gts = read_json(args.gtpath)["annotations"]
    vid2gts = {}
    for jterm in gts:
        vid2gts[jterm["image_id"]] = jterm
    # example in output file
    # {
    #     "query_idx": 
    #         {
    #             "timestamp": [47.0, 60.0],
    #             "query": "a person is shown tying a plant into a bun.",
    #             "vid": "xHr8X2Wpmno.mp4"
    #         }, 
    #     ...
    # }
    fmt_datas = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = jterm["id"]
        highlights, clipscores = format_vhd_output(gcap, vid2gts[vid])
        if len(highlights) == 0:
            cnt += 1
            print(vid, query+"\n", gcap+"\n")
            # pdb.set_trace()
        else:
            # print(gcap)
            # print(timestamps)
            pass
        result = {}
        result["qid"] = qid
        result["query"] = query
        result["vid"] = vid
        result["pred_saliency_scores"] = clipscores
        fmt_datas.append(result)

    print(f'parse failed number: {cnt}')
    split = args.inpath.split('/')[-1].split('_')[0]
    out_file = args.inpath.split('/')[-2]
    out_path = f'{out_file}_{split}.json'
    if args.outpath != '':
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(args.outpath, out_path)
        write_json(os.path.join(os.getcwd(), out_path), fmt_datas)
    else:
        infile = args.inpath.split('/')[-1]
        outfile = "fmt_" + infile
        out_path = args.inpath.replace(infile, outfile)
        write_json(out_path, fmt_datas)