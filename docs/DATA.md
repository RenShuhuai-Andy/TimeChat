## Download
- YouCook2: http://youcook2.eecs.umich.edu/download
- Charades-STA: https://github.com/jiyanggao/TALL#charades-sta-anno-download
- QVHighlight: https://github.com/jayleicn/moment_detr/blob/main/data/README.md
- todo

The file structure looks like:
```
data/
|–– YouCook2-BB/
    |-- YouCook2_asr_denseCap/
        |-- youcook2_6fps_224/
            |-- psVc_8RL1ow.mp4
            |-- ...
        |-- instruct_1.2k_dvc_youcook2.json
|–– Charades/
    |-- videos/
        |-- FIAJP.mp4
        |-- ...
    |-- instruct_tvg_12.4k_charades.json
|–– QVhighlights/
    |-- videos/
        |-- train
            |-- v_zzWIB6kuuAQ_210.0_360.0.mp4
            |-- ...
        |-- val
            |-- v_ZxHh_2YdmT4_60.0_210.0.mp4
            |-- ...
    |-- instruct_vhd_6.9k_qvhighlights.json
```