PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset kuairand-1k --mode eval
PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset movielens-1m --mode eval
PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset movielens-20m --mode eval
PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan --mode eval

tensorboard --logdir=logs/


# Train msan dataset using 8 GPUs
WORLD_SIZE=8 PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan --mode train
# Basic iterable dataset
WORLD_SIZE=8 PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --mode train

PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --world_size 8 --data_path /scratch/azureml/cr/j/fa73d0e8d8ee4b5eb8e8bb9145808c48/cap/data-capability/wd/INPUT_data/

# Iterable dataset with buffered shuffle
WORLD_SIZE=8 PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable-buffered --mode train



# /home/aiscuser commands,
conda init
# re-open terminal
# read the {environment path} from the first line, should be like:
# /azureml-envs/azureml_efe2b52c31ee0fe3011abc050798ff87
conda activate /azureml-envs/azureml_efe2b52c31ee0fe3011abc050798ff87

# input data mount path
/scratch/azureml/cr/j/{ssh_id}/cap/data-capability/wd/INPUT_data/
/scratch/azureml/cr/j//04bb871dd38a4b63bdab0d99e74b0c69/cap/data-capability/wd/INPUT_data/

PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --data_path /scratch/azureml/cr/j/55428fd652604523af0a40ae36ce2d22/cap/data-capability/wd/INPUT_data

PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --world_size 8 --data_path /scratch/azureml/cr/j/55428fd652604523af0a40ae36ce2d22/cap/data-capability/wd/INPUT_data

nohup bash -c 'PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --data_path /scratch/azureml/cr/j/55428fd652604523af0a40ae36ce2d22/cap/data-capability/wd/INPUT_data' > train_$(date +%m%d_%H%M).log 2>&1 &

nohup bash -c 'CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --data_path /scratch/azureml/cr/j/55428fd652604523af0a40ae36ce2d22/cap/data-capability/wd/INPUT_data --mode eval' > logs/train_1w_0001_save_checkpoint_debug.log 2>&1 &

nohup bash -c 'PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --world_size 8 --data_path /scratch/azureml/cr/j/7a7a7a93f2044939950c3b0f94b0c5bb/cap/data-capability/wd/INPUT_data --exp_name train_5m_0001_30epoch_batch256_mlp1_8stu_seqlen256' > logs/train_5m_0001_30epoch_batch256_mlp1_8stu_seqlen256.log 2>&1 &


# nohup bash -c 'PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --world_size 8 --data_path ${{inputs.data}}' > logs/train_5m_001_8epoch_batch512_mlp1-8gpu.log 2>&1 &



# echo "Starting training at $(date)"
# PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --world_size 8 --data_path ${{inputs.data}} 2>&1 | tee logs/train_5m_0001_100epoch_batch128_mlp1_multiply_64stu.log
# echo "Training completed at $(date), starting evaluation"
# PYTHONPATH=. python dlrm_v3/train/train_ranker.py --dataset msan-iterable --world_size 8 --data_path ${{inputs.data}} --mode eval 2>&1 | tee logs/eval_5m_0001_100epoch_batch128_mlp1_multiply_64stu.log
