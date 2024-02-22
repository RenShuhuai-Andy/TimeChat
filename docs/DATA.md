# Download

## Video & Annotations
- YouCook2: http://youcook2.eecs.umich.edu/download
- Charades-STA: https://github.com/jiyanggao/TALL#charades-sta-anno-download
- QVHighlight: https://github.com/jayleicn/moment_detr/blob/main/data/README.md
- ActivityNet Captions: http://activity-net.org/download.html
- ViTT: https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT
- DiDeMo: https://github.com/LisaAnne/LocalizingMoments?tab=readme-ov-file#dataset
- QuerYD: https://www.robots.ox.ac.uk/~vgg/data/queryd/
- HiREST: https://github.com/j-min/HiREST
- TVSum: https://github.com/yalesong/tvsum
- SumMe: http://classif.ai/dataset/ethz-cvl-video-summe/
- COIN: https://github.com/coin-dataset/annotations
- YT-Temporal: https://rowanzellers.com/merlot/#data

### Compressing Videos
We preprocess videos to lower FPS and dimension to reduce storage and to improve data loading. For videos, you may use
```bash
ls -U /path/to/raw/video >> /path/to/video_names.txt

# for YouCook2
python utils/compress_video_data.py \
--input_root=/path/to/raw/videos/ \
--output_root=data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/ \
--input_file_list_path=/path/to/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 24

# for ActivityNet Captions
python utils/compress_video_data.py \
--input_root=/path/to/raw/videos/ \
--output_root=data/Activitynet_Captions/anet_6fps_224 \
--input_file_list_path=/path/to/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 24
```

Note that the audio is also removed from the video files.

### Process HiREST
For `HiREST_step` dataset in TimeIT (`instruct_action_0.5k_hirest.json`), you can use the following commands to crop the videos into clips:
1. download the videos directory from HiREST (see the following file structure)
2. download the annotations directory from https://huggingface.co/datasets/ShuhuaiRen/TimeIT/tree/main/data/step_localization/hirest_step
3. run `python utils/process_hirest.py`

### Process Valley
1. download the videos from Valley (https://github.com/RupertLuo/Valley/blob/main/Crawler/README.md)
2. run `python utils/process_valley.py` (including video cropping for VATEX)

### Automatic speech transcription
To use the automatic speech transcription, you can use the following commands:
1. Extract audios from videos. Examples:
    ```bash
    python utils/extract_audio.py --dir /home/v-shuhuairen/mycontainer/data/DiDeMo/ --video_folder videos
    for i in {0..179}; do python utils/extract_audio.py --dir /home/v-shuhuairen/mycontainer/data/COIN/ --video_folder videos_ali/${i} --audio_folder audio_files/${i}; done
    ```
2. Automatic speech transcription using [Whisper](https://github.com/linto-ai/whisper-timestamped). Examples:
    ```bash
    python utils/asr/asr.py --audio_dir audio_files --dir /home/v-shuhuairen/mycontainer/data/DiDeMo/
    for i in {0..179}; do python utils/asr/asr.py --dir /home/v-shuhuairen/mycontainer/data/COIN/ --audio_dir audio_files/${i}; done
    ```

3. Clean the ASR results
    ```bash
    python utils/clean_asr.py --dir /home/v-shuhuairen/mycontainer/data/DiDeMo/whisper_outputs_with_time/
    ```

4. Incorporate the ASR results into the instructions
    ```bash
    python utils/asr_data_process.py
    ```

## Instruction data
- TimeIT: https://huggingface.co/datasets/ShuhuaiRen/TimeIT
- Valley: https://huggingface.co/datasets/luoruipu1/Valley-Instruct-65k

Clone the TimeIT dataset under `TimeChat/data`
```bash
git clone https://huggingface.co/datasets/ShuhuaiRen/TimeIT
```

The file structure looks like:
```
data/
|–– TimeIT/

# for evaluation
|–– YouCook2-BB/
    |-- YouCook2_asr_denseCap/
        |-- youcook2_6fps_224/
            |-- psVc_8RL1ow.mp4
            |-- ...
        |-- instruct_dvc_1.2k_youcook2.json
        |-- train.caption_coco_format.json
        |-- val.caption_coco_format.json
        |-- test.caption_coco_format.json
|–– Charades/
    |-- videos/
        |-- FIAJP.mp4
        |-- ...
    |-- charades_annotation/
        |-- train.caption_coco_format.json
        |-- test.caption_coco_format.json
    |-- instruct_tvg_12.4k_charades.json
|–– QVhighlights/
    |-- videos/
        |-- train
            |-- v_zzWIB6kuuAQ_210.0_360.0.mp4
            |-- ...
        |-- val
            |-- v_ZxHh_2YdmT4_60.0_210.0.mp4
            |-- ...
    |-- annotations_raw/
        |-- highlight_train_release.jsonl
        |-- highlight_val_release.jsonl
        |-- highlight_test_release.jsonl
    |-- instruct_vhd_6.9k_qvhighlights.json
    
# for instruction tuning (from TimeIT)
|-- Activitynet_Captions/
    |-- anet_6fps_224/
        |-- v_---9CpRcKoU.mp4
        |-- ...
|-- vitt/
    |-- raw_videos/
        |-- --g9p1A_fF0.mp4
        |-- ...
|-- DiDeMo/
    |-- videos/
        |-- 10015567@N08_3655084291_d8b58466fa.mp4
        |-- ...
|–– QuerYD/
    |-- QuerYD_downloader/
        |-- videos/
            |-- video---YU8YcWeUU
            |-- ...
|-- HiREST/
    |-- clips/
        |-- -17SPG-3Jis_53_242.mp4
        |-- ...
    |-- videos/
        |-- -17SPG-3Jis.mp4
        |-- ...
|-- TVSum/
    |-- videos/
        |-- 0tmA_C6XwfM.mp4
        |-- ...
|-- SumMe/
    |-- videos/
        |-- Air_Force_One.mp4
        |-- ...
|-- COIN/
    |-- videos_ali/
        |-- 0/
            |-- -8NaVGEccgc.mp4
            |-- ...
        |-- ...
|-- yttemporal180m/
    |-- videos/
        |-- --LGz4Kb3AA.mp4
        |-- ...

# for instruction tuning (from valley)
|-- vatex/
    |-- videos/
        |-- --07WQ2iBlw_000001_000011.mp4
        |-- ...
|-- jukin/
    |-- videos/
        |-- v_1000045.mp4
        |-- ...
```

### Process Custom Instruction-Tuning Data
1. If you want to construct instruction-tuning datasets by yourself, you can refer to the processing files under `utils/construct_it_data`.
2. If you want to use a subset of the TimeIT dataset, or incorporate more datasets, you can modify `file_to_merge` in `utils/construct_timeit_data/merge.py` and rerun it.

The datum schema should be like:
```json
{
  "video": "ActivityNet_asr_denseCap/anet_6fps_224/v_MinmayCk2Nk.mp4", 
  "QA": [
    {
      "q": "Capture and describe the activity events in the given video, specifying their respective time intervals, and outputting the time intervals in the 'start - end seconds format'.", 
      "a": "0.0 - 9.1 seconds, We see a male gymnast prepare to use the pommel horse.  9.1 - 10.8 seconds, The man mounts the pommel horse and spins his legs around it.  29.9 - 35.2 seconds, The man does a handstand and dismounts.  35.7 - 38.6 seconds, The man takes a bow and starts walking away."
    }
  ], 
  "source": "Activitynet_Captions"
}
```

### Process Custom Evaluation Data into COCO Format
1. If you want to construct evaluation datasets by yourself and use our eval code, you need to transform data annotations into coco format. You can write processing code in `utils/get_coco_format.py`.

The datum schema should be like:

for **dense video captioning**:
```json
{
    "annotations": 
    [
        {
            "image_id": "3MSZA.mp4",
            "duration": 206.86,
            "segments": [[47, 60], [67, 89], [91, 98], [99, 137], [153, 162], [163, 185]],
            "caption": "pick the ends off the verdalago. ..."
        },
        ...
    ]
}
```
for **temporal video grounding**:
```json
{
    "annotations": 
    [
      {   
            "image_id": "3MSZA.mp4", 
            "caption": "person turn a light on.",
            "timestamp": [24.3, 30.4]
      },
      ...
    ]
}
```