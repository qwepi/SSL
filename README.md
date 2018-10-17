# SSL 

Source code and pretrained model for 24th Asia and South Pacific Design Automation Conference paper: 

Semi-Supervised Hotspot Detection with Self-Paced Multi-Task Learning.

Author: Ying Chen, Yibo Lin, Tianyang Gai, Yajuan Su, Yayi Wei, and David Z. Pan

## Dataset

Feature Tensor Extraction Data is already within this repo, original images can be found at http://appsrv.cse.cuhk.edu.hk/~hyyang/files/iccad-official.tgz

## Dependencies

numpy, tensorflow (tested on 1.5), pandas, json, ConfigParser, progress

## Test on Released Models

e.g. to test iccad2 with 10% labeled samples (random seed = 50) on the released model, you need to modify iccad2\_config.ini

set ```model_path=./models/iccad2/unlossfix_SSL_m10000_p0.1s50/model-p0.1-s50-step9999.ckpt```  

set ```train_ratio=0.1``` 

set ```seed=50``` 

set ```b=2``` and

```python test_SSL.py iccad2_config.ini```

## Train

e.g. to train iccad2 with 10% labeled samples (random seed =50), you need to modify iccad2\_config.ini

set ```save_path=./models/iccad2/ssl/```

set ```train_ratio=0.1``` 

set ```seed=50```

set ```b=2``` and

```python train_SSL.py iccad2_config.ini```

## Test

e.g. to test iccad2, you need to modify iccad2\_config.ini

set ```model_path=./models/iccad2/ssl/model.ckpt```(note: This model path should be where you save your model when training)  

set ```train_ratio=0.1``` 

set ```seed=50```

set ```b=2``` and

```python test_SSL.py iccad2_config.ini```

## Batch Process

e.g. to train and test iccad2 with 10%, 30%, 50% labeled samples and different random seeds(50,100,150), you need to modify run.sh as folows:

for b in 2: do

for train_p in 0.1 0.3 0.5; do

for seed in 50 100 150; do

and 

```source run.sh```

then when all the runnings are done, go to folder "log_SSL" to check the testing results.

## Acknowledgement

The code is based on [Haoyu Yang's source code](https://github.com/phdyang007/dlhsd), thanks for his sharing.

