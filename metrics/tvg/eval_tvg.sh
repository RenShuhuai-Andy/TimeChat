# default: eval all instances according to the gt_file
python eval_tvg.py --pred_file your_pred_file --gt_file your_gt_file 


# use --sample
# eval sampled instances according to the pred_file
# e.g. # gt examples:500, # pred examples:50 -> # eval examples:50
python eval_tvg.py --pred_file your_pred_file --gt_file your_gt_file  --sample