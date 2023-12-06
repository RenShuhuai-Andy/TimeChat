# example in pred file
# {
#     "xHr8X2Wpmno.mp4": [
#         {
#             "timestamp": [47.0, 60.0],
#             "caption": "a person is shown tying a plant into a bun and putting the bun into a pink jar."
#         }, 
#         ...
#     ]
#     ...
# }

pred_file='/home/yaolinli/code/Ask-Anything/video_chat/results/eval_7b_instruct111k_timeit-vally-llava-66k_bz8_f8_epoch3_youcook.json'
gt_file='/home/yaolinli/dataset/YouCook2_asr_denseCap/val.caption_coco_format.json'

# pred_file='/home/yaolinli/code/Ask-Anything/video_chat/results/eval_7b_instruct11.2k_youcook2-anet_bz8_f8_epoch3_anet.json'
# gt_file='/home/yaolinli/dataset/ActivityNet_asr_denseCap/val.caption_coco_format.json'

python eval_dvc.py --pred_file $pred_file --gt_file $gt_file   --analyze

