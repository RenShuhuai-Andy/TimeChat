import json
import random


def read_json(path):
    with open(path, "r") as fin:
        data = json.load(fin)
    return data


file_to_merge = [
    '../data/TimeIT/data/transcribed_speech_generation/yttemporal/instruct_tsg_31.6k_yttemporal.json',
    '../data/TimeIT/data/step_localization/coin/instruct_action_9.0k_coin.json',
    '../data/TimeIT/data/step_localization/hirest_step/instruct_action_0.5k_hirest.json',
    '../data/TimeIT/data/temporal_video_grounding/queryd/instruct_tvg_14.6k_queryd.json',
    '../data/TimeIT/data/temporal_video_grounding/hirest/instruct_tvg_0.5k_hirest.json',
    '../data/TimeIT/data/temporal_video_grounding/didemo/instruct_tvg_33.0k_didemo.json',
    '../data/TimeIT/data/dense_video_captioning/anet/instruct_dvc_10.0k_anet.json',
    '../data/TimeIT/data/dense_video_captioning/vitt/instruct_dvc_5.1k_vitt.json',
    '../data/TimeIT/data/video_summarization/summe/instruct_vhd_25_summe.json',
    '../data/TimeIT/data/video_summarization/tvsum/instruct_vhd_50_tvsum.json',
]

merge_data = []
for fi, fpath in enumerate(file_to_merge):
    data = read_json(fpath)
    for i, jterm in enumerate(data):
        data[i]["source"] = file_to_merge[fi].split("/")[-2]
    merge_data.extend(data)
    
random.shuffle(merge_data)

out_path = "../data/TimeIT/data/time/instruct_time-sensitive_{}k_asr.json".format(round(len(merge_data)/1000), 1)
print("save merge data at {}".format(out_path))
with open(out_path, "w") as fout:
    json.dump(merge_data, fout)
