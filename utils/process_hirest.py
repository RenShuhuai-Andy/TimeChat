import json
import os
import pdb
from pathlib import Path
from tqdm import tqdm
from pathlib import Path


def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def write_json(path, data):
    with open(path, "w") as fout:
        json.dump(data, fout)


def get_clip(video_root, clip_root, vid, clip_id, start, end):
    video_path = os.path.join(video_root, vid)
    clip_path = os.path.join(clip_root, clip_id)
    if not os.path.exists(clip_root):
        # mkdir 
        Path(clip_root).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(clip_path):
        os.system(f"ffmpeg -i {video_path} -ss {start} -to {end} -c copy {clip_path}")
    return


if __name__ == "__main__":
    data_path = "TimeIT/data/step_localization/hirest_step/annotations/"
    video_root = "HiREST/videos/"
    clip_root = "HiREST/clips/"
    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_path, f"all_data_{split}.json")
        anno = read_json(file_path)

        grounding_data = []
        step_data = []
        gid = 0
        sid = 0
        for query, qdatas in tqdm(anno.items()):
            for vid, vdata in qdatas.items():
                if not vdata["relevant"] or not vdata["clip"]:
                    continue
                v_duration = vdata["v_duration"]
                bounds = vdata["bounds"]
                steps = vdata["steps"]
                if len(steps) == 0:
                    continue
                # grounding data
                grd_jterm = {}
                grd_jterm["image_id"] = vid
                grd_jterm["caption"] = query
                grd_jterm["id"] = gid
                gid += 1
                grd_jterm["timestamp"] = bounds
                grd_jterm["duration"] = v_duration
                grounding_data.append(grd_jterm)
                # step data
                clip_id = vid.split(".mp4")[0] + "_" + str(bounds[0]) + "_" + str(bounds[1]) + ".mp4"
                # extract clip from original video
                get_clip(video_root, clip_root, vid, clip_id, bounds[0], bounds[1])
                # prepare step data
                step_jterm = {}
                step_jterm["image_id"] = clip_id
                segments = []
                labels = []
                for step in steps:
                    cap = step["heading"]
                    abs_time = step["absolute_bounds"]
                    timestamp = [abs_time[0] - bounds[0], abs_time[1] - bounds[0]]
                    segments.append(timestamp)
                    labels.append(cap)
                step_jterm["segments"] = segments
                step_jterm["labels"] = labels
                step_jterm["id"] = sid
                step_jterm["duration"] = bounds[1] - bounds[0]
                sid += 1
                step_data.append(step_jterm)

        if not os.path.exists("grounding_anno"):
            Path("grounding_anno").mkdir(parents=True, exist_ok=True)
        if not os.path.exists("step_anno"):
            Path("step_anno").mkdir(parents=True, exist_ok=True)
        write_json(f"grounding_anno/{split}.json", grounding_data)
        write_json(f"step_anno/{split}.json", step_data)
        print(f"[{split}] grounding data: {len(grounding_data)}, step data: {len(step_data)}")
