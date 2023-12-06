# Calculate Metrics 

#### Dense video captioning task 

```
cd dvc/
```

Set the `pred_file` and `gt_file` and run:

```
python eval_dvc.py --pred_file $pred_file --gt_file $gt_file
```

Evaluation for paragraph captioning, run:

```
python eval_dvc.py --pred_file $pred_file --gt_file $gt_file --paragraph
```

#### Temporal video grounding task

```
cd tvg/
```

Set the `pred_file` and `gt_file` and run:

```
python eval_dvc.py --pred_file $pred_file --gt_file $gt_file
```


#### Video highlight detection task

```
cd vhd/
```

Set the `pred_file` and `gt_file` and run:

```
python eval_highlights.py --pred_file $pred_file --gt_file $gt_file
```