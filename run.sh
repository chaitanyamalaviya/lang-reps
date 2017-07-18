#!/bin/sh
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=100g
#SBATCH -t 0 

module load cuda-8.0 cudnn-8.0-5.1

source activate dynet
export DYLD_LIBRARY_PATH=/home/cmalaviy/dynet/build/dynet/:$DYLD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cuda-8.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/cudnn-8.0/include:$CPATH
export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH

# BIBLE MODEL
# python -u trainer.py --dynet-mem 11500 --dynet-gpu --model_type=attention --minibatch_size=32 --reader_mode=parallel --train_src=/projects/tir2/users/cmalaviy/bible-corpus/train_src_lang.txt --train_tgt=/projects/tir2/users/cmalaviy/bible-corpus/train_tgt_en.txt --valid_src=/projects/tir2/users/cmalaviy/bible-corpus/val_src_lang.txt --valid_tgt=/projects/tir2/users/cmalaviy/bible-corpus/val_tgt_en.txt --trainer=adam --load=bible_model_final.checkpoint --log_output=bible_model_final_appended.log

# EXTRACT LVS
#python -u trainer.py --extract_lvs=lang_vecs_bible.npy --dynet-mem 11500 --dynet-gpu --model_type=attention --minibatch_size=32 --reader_mode=parallel --train_src=test-langs/train.en-de.low.de --train_tgt=test-langs/train.en-de.low.en --valid_src=/projects/tir2/users/cmalaviy/bible-corpus/val_src_lang.txt --valid_tgt=/projects/tir2/users/cmalaviy/bible-corpus/val_tgt_en.txt --trainer=adam --load=bible_model_final.checkpoint
