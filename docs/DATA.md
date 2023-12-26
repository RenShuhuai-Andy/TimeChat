## Download

### Video & Annotations
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

### Instruction data
- TimeIT: https://huggingface.co/datasets/ShuhuaiRen/TimeIT
- Valley: https://huggingface.co/datasets/luoruipu1/Valley-Instruct-73k

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

## Processed Custom Data
1. If you want to construct instruction-tuning datasets by yourself, you can refer to the processing files under `utils/construct_it_data`.
2. If you want to use a subset of the TimeIT dataset, or incorporate more datasets, you can modify `file_to_merge` in `utils/construct_timeit_data/merge.py` and rerun it.