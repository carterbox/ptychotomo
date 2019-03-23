#!/bin/bash

#SBATCH -t 2-00:00:00
#SBATCH -J ptychotomo
#SBATCH -p v100
#SBATCH --mem 120G
#SBATCH -o p_rec_gpu.out
#SBATCH -e p_rec_gpu.err

module add cuda_x86

source ~/.bashrc
source activate py35


python -u lcurvechip.py 0 >res0 &
python -u lcurvechip.py 1 >res1 &
python -u lcurvechip.py 2 >res2 &
python -u lcurvechip.py 3 >res3 &
wait
