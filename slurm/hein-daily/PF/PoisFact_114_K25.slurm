#!/bin/bash
#SBATCH --job-name=PF_hein-daily # short name for your job
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task
#SBATCH -o /home/jvavra/STBS/out/hein-daily/%x_%j_%N.out      # save stdout to file. The filename is defined through filename pattern
#SBATCH -e /home/jvavra/STBS/err/hein-daily/%x_%j_%N.err      # save stderr to file. The filename is defined through filename pattern
#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)

. /opt/apps/2023-04-11_lmod.bash
ml purge
ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn

cd /home/jvavra/STBS/
conda activate env_TBIP

python /home/jvavra/STBS/code/poisson_factorization.py  --data=hein-daily  --num_topics=25  --addendum=114
