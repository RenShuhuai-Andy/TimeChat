import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/home/v-shuhuairen/mycontainer/data/Charades/whisper_outputs_with_time/')
parser.add_argument('--src_folder', type=str, default='tiny.en')
parser.add_argument('--tgt_folder', type=str, default='tiny.en.cleaned')
args = parser.parse_args()

src_dir = os.path.join(args.dir, args.src_folder)
tgt_dir = os.path.join(args.dir, args.tgt_folder)

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

files = sorted(list(os.listdir(src_dir)))
for file in tqdm(files):
    if not os.path.exists(f'{tgt_dir}/{file}'):
        print(file)
        with open(f'{src_dir}/{file}', 'r') as fr:
            lines = fr.readlines()

        segments = []
        last_text = ''
        for line in lines:
            s, e, text = line.split('\t')
            s, e = float(s), float(e)
            text = text.strip()
            # print((s,e,text))
            if len(text.split()) < 3:
                continue
            if text == last_text:
                continue
            last_text = text
            segments.append((s, e, text))
        with open(f'{tgt_dir}/{file}', 'w') as fw:
            for segment in segments:
                s, e, text = segment
                fw.write(f'{s:.2f}\t{e:.2f}\t{text}\n')
