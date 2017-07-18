#!/bin/sh
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH -t 0

module load cuda-8.0 cudnn-8.0-5.1

source activate dynet
export DYLD_LIBRARY_PATH=/home/cmalaviy/dynet/build/dynet/:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/cudnn-8.0/include:$CPATH
export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH

python -u trainer.py --dynet-mem 11700 --dynet-gpu --eval --model_type=attention --train_src=test-svo2/comb_src.txt --train_tgt=test-svo2/comb_tgt.txt --valid_src=/projects/tir2/users/cmalaviy/bible-corpus/val_src_lang.txt --valid_tgt=/projects/tir2/users/cmalaviy/bible-corpus/val_tgt_en.txt --test_src=test-svo2/comb_src.txt --test_tgt=test-svo2/comb_tgt.txt --trainer=adam --load=bible_model_final.checkpoint
