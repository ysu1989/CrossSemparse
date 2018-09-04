## run script with specific GPU
mode=$1
gpu=$2
shift
shift
CUDA_VISIBLE_DEVICES=${gpu} python scripts/${mode}.py $@

