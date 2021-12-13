# YOLO v5 prune project
## first step: trasfer training: 
train on another smaller one class dataset(transfer to a new data set):
```python
python3 train.py --weights ../weights_yolov5/yolov5s.pt --cfg models/yolov5s.yaml --data data/hand.yaml --epoch 40 --name <<output_dir_name>> --project <<output_dir>> --batch-size 16 --img 640 
```

## second step: sparse training:
apply L1 norm regulization on BN's loss, and make the weight sparse:
```python
python3 train.py --weights <<path_to_pt>> --st --sr 0.05  --data data/hand.yaml --epoch 20 --name <<output_dir_name>> --project <<output_dir>> --batch-size 16 --img 640 
```
## third step: prune model: 
Pruning:
```python
python prune.py --weights <<path_to_pt>> --percent 0.4 --data data/coco_hand.yaml --name <<output_dir_name>> --project <<output_dir>> 
```
## fourth step: fine tune: 
(adding half when output):
```python
python3 train.py --weights <<path_to_pt>> --data data/hand.yaml --epoch 20 --name <<output_dir_name>> --project <<output_dir>> --batch-size 16 --img 640 --half
```
## fifth step: Sparse-Prune-Fine tune loop: 
repeat by training sparse - prune - fine tune again.

## seventh step: evaluation:
evaluation:
```python
python val.py --weight <<path_to_pt>> --project <<output_dir>> --name <<output_dir_name>> --imgsz 640 --data data/hand.yaml --device cuda:0 --half
```
## sixth step: export to desired type:
export:
for example: tensorrt .engine file:
```python
python export.py --weight <<path_to_pt>> --imgsz 640 --data data/hand.yaml --include engine --device cuda:0 --half
```
file would be exported to the path where the pt is.

