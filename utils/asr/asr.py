import os
import whisper_timestamped as whisper
# pip3 install git+https://github.com/linto-ai/whisper-timestamped
from tqdm import tqdm
import argparse
import os.path as osp
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='tiny.en')
parser.add_argument('--audio_dir', type=str, default='audio_files')
parser.add_argument('--dir', type=str, default='/home/v-shuhuairen/mycontainer/data/yttemporal180m/')
args = parser.parse_args()

model = whisper.load_model(args.model_name)
postfix = args.audio_dir.split('/')[-1] if '/' in args.audio_dir else None  # for COIN
if postfix is not None:
    output_dir = osp.join(args.dir, f'whisper_outputs_with_time/{args.model_name}/{postfix}')
else:
    output_dir = osp.join(args.dir, f'whisper_outputs_with_time/{args.model_name}')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for path in tqdm(sorted(list(os.listdir(osp.join(args.dir, args.audio_dir))))):
    id = path.split('.')[0]
    if os.path.exists(f'{output_dir}/{id}.txt'):
        continue
    try:
        result = model.transcribe(osp.join(args.dir, f'{args.audio_dir}/{path}'))
    except:
        print(f'error in {path}')
        continue

    seg_start = 0
    seg_end = 0
    seg_text = ''
    first_part_flag = True
    with open(f'{output_dir}/{id}.txt', 'w') as fw:
        for seg in result["segments"]:
            s, e, text = seg['start'], seg['end'], seg['text'].strip()
            fw.write(f'{s:.2f}\t{e:.2f}\t{text}\n')
