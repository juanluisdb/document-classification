#!/bin/bash
#SBATCH --job-name b0-1
#SBATCH -D /home/bsc31/bsc31282/document-classification/text_model_updated
#SBATCH --output /home/bsc31/bsc31282/document-classification/text_model_updated/%j.out
#SBATCH --error /home/bsc31/bsc31282/document-classification/text_model_updated/%j.err
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --ntasks 1
#SBATCH -c 40
#SBATCH --time 03:00:00

module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML arrow/3.0.0 torch/1.9.0a0 text-mining/2.1.0
export PYTHONUNBUFFERED=1

python main.py --json_file '/home/bsc31/bsc31282/document-classification/text_model_updated/configs/config.json'

python ensemble.py --json_file '/home/bsc31/bsc31282/document-classification/text_model_updated/configs/config_ensemble.json'
