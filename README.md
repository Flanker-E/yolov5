# YOLO v5 prune project
## first step: trasfer training: 
train on another smaller one class dataset
## second step: sparse training:
apply L1 norm regulization on BN's loss, and make the weight sparse

