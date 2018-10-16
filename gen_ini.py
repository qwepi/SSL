
'''
generate configure file (e.g. iccad2_config.ini) automatically

'''

import sys

initial_ini = """
[dir]
#benchmark path for training set
train_path = ./benchmarks/BENCHMARK/train
#benchmark path for model testing
test_path  = ./benchmarks/BENCHMARK/test

#path to save model
#save_path  = ./models/BENCHMARK/unlossfix_SSL_m10000_p0.1_seed50/
save_path = SAVE_PATH

#path to trained model
#model_path = ./models/BENCHMARK/unlossfix_SSL_m10000_p0.1_seed50/model-p0.1-s50-step9999.ckpt
model_path = MODEL_PATH

[feature]
#the length of feature tensor
ft_length  = 32
block_dim  = 12

train_ratio = TRAIN_RATIO
seed = SEED 
b = bB
"""

b = int(sys.argv[1])
train_p = float(sys.argv[2])
seed = int(sys.argv[3])
ini_filename = str(sys.argv[4])

ini = initial_ini.replace("BENCHMARK", "iccad%d" % (b))
ini = ini.replace("SAVE_PATH", "models/iccad%d/release/SSL_p%g_seed%d/" % (b, train_p, seed))
ini = ini.replace("MODEL_PATH", "models/iccad%d/release/SSL_p%g_seed%d/model-p%g-s%d-step9999.ckpt" % (b, train_p, seed,train_p,seed))
ini = ini.replace("TRAIN_RATIO", "%g" % (train_p))
ini = ini.replace("SEED", "%d" % (seed))
ini = ini.replace("bB", "%d" % (b))

with open(ini_filename, "w") as f:
    f.write(ini)
