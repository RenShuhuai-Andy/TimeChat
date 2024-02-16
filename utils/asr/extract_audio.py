import os
import os.path as osp
import json
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/home/v-shuhuairen/mycontainer/data/yttemporal180m/')
parser.add_argument('--video_folder', type=str, default='raw_videos')
parser.add_argument('--audio_folder', type=str, default='audio_files')
args = parser.parse_args()

video_folder = osp.join(args.dir, args.video_folder)
audio_folder = osp.join(args.dir, args.audio_folder)
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)
videos = os.listdir(video_folder)

for video in tqdm(videos):
    idx = video.split('.')[0]
    if not os.path.exists(f'{audio_folder}/{idx}.mp3'):
        cmd = f'ffmpeg -i {video_folder}/{video} -q:a 0 -map a {audio_folder}/{idx}.mp3 '
        print(cmd)
        os.system(cmd)
