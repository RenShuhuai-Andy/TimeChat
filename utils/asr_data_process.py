import json
import os
from tqdm import tqdm
from copy import deepcopy

folders = [
        'data/COIN/',
        'data/QuerYD/QuerYD_downloader/',
        'data/DiDeMo/',
        'data/Activitynet_Captions/',
        'data/vitt/',
        'data/SumMe/',
        'data/TVSum/',

]

filenames = [
         'instruct_action_9.0k_coin.json',
         'instruct_tvg_14.6k_queryd.json',
         'instruct_tvg_33.0k_didemo.json',
         'instruct_dvc_10.0k_anet.json',
         'instruct_dvc_5.1k_vitt.json',
         'instruct_vhd_25_summe.json',
         'instruct_vhd_50_tvsum.json',
]

for folder, filename in zip(folders, filenames):
    with open(os.path.join(folder, filename), 'r') as f:
        data = json.load(f)

    max_num_asr = 15  # only use max to 20 asr
    data_with_asr = []
    asr_folder = os.path.join(folder, 'whisper_outputs_with_time/tiny.en.cleaned/')
    for d in tqdm(data):
        new_d = deepcopy(d)
        video_name = d['video'].split('/')[-1].split('.')[0]
        if 'coin' in folder.lower():
            class_id = d['video'].split('/')[-2]
            asr_path = os.path.join(asr_folder, class_id, video_name + '.txt')
        if 'qvhighlights' in folder.lower():
            asr_path = os.path.join(asr_folder, 'train', video_name + '.txt')
        else:
            asr_path = os.path.join(asr_folder, video_name + '.txt')
        if os.path.exists(asr_path):
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
        else:
            final_asr = 'None.'
        for i in range(len(d['QA'])):
            new_d['QA'][i]['q'] = f"Transcribed speech: {final_asr} Based on the video content and possible transcribed speech, {d['QA'][i]['q']}"
        data_with_asr.append(new_d)

    assert len(data) == len(data_with_asr)

    with open(os.path.join(folder, filename.replace('.json', f'_{max_num_asr}asr.json')), 'w') as f:
        f.write(json.dumps(data_with_asr))
