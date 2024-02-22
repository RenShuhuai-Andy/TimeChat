import json
import pdb
import os
from pathlib import Path
from tqdm import tqdm


def get_clip(video_root, clip_root, vid, clip_id, start, end):
    video_path = os.path.join(video_root, vid)
    clip_path = os.path.join(clip_root, clip_id)
    if not os.path.exists(video_path):
        print(f"video {video_path} not exist!")
        return
    if not os.path.exists(clip_root):
        # mkdir
        Path(clip_root).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(clip_path):
        # os.system(f"ffmpeg -i {video_path} -ss {start} -to {end} -c copy {clip_path}")
        command = ['ffmpeg',
                   '-i', video_path,
                   '-ss', start,
                   '-t', end,
                   '-c:v', 'libx264', '-c:a', 'copy',
                   '-threads', '1',
                   # '-loglevel', 'panic',
                   clip_path]
        command = ' '.join(command)
        os.system(command)
    return


# path = 'Valley_instruct_73k.json'
path = '/home/v-shuhuairen/mycontainer/data/Valley-Instruct-65k/valley_instruct_65k.json'
with open(path, 'r') as fin:
    data = json.load(fin)


sources = {}
exist_video_data_num = {
    'VATEX': 0,
    'jukinmedia': 0
}
video_root = {
    'raw_VATEX': '/path/to/vatex/raw_videos/',  # before cropping
    'VATEX': '/path/to/vatex/videos/',  # after cropping
    'jukinmedia': '/path/to/jukin/videos/'
}

instruct_data = []
for item in tqdm(data):
    vsource = item['source']
    if vsource == 'VATEX':
        vid = item['v_id'] + '.mp4'
        output_filename = os.path.join(video_root[vsource], vid)
        if os.path.exists(output_filename):  # exist cropped VATEX videos
            continue
        # crop VATEX videos
        raw_video_name = item['video']
        start_time = vid[12:18]
        end_time = vid[19:25]
        get_clip(video_root=video_root['raw_VATEX'], clip_root=video_root[vsource], vid=raw_video_name, clip_id=vid,
                 start=start_time, end=end_time)
    elif vsource == 'jukinmedia':
        vid = item['video']
    else:
        print("not existing resource!")
    # filter data samples do not have videos
    if not os.path.exists(os.path.join(video_root[vsource], vid)):
        continue
    
    exist_video_data_num[vsource] += 1
    jterm = {}
    jterm["video"] = os.path.join(video_root[vsource], vid)
    convs = []
    for ti in range(0, len(item["conversations"]), 2):
        turn_human = item["conversations"][ti]
        turn_gpt = item["conversations"][ti+1]
        assert turn_human["from"] == "human"
        assert turn_gpt["from"] == "gpt"
        qa = {}
        qa["q"] = turn_human["value"]
        qa["q"] = qa["q"].replace("<video>\n", "")
        qa["q"] = qa["q"].replace("\n<video>", "")
        qa["a"] = turn_gpt["value"] 
        convs.append(qa)
    jterm["QA"] = convs
    instruct_data.append(jterm)

print(exist_video_data_num)
valid_num = sum(list(exist_video_data_num.values()))
print("# samples have videos {}; # total samples {}".format(valid_num, len(data)))


# write json
with open("instruct_valley_{}k.json".format(round(valid_num/1000), 1), "w") as fout:
    json.dump(instruct_data, fout, indent=4)
