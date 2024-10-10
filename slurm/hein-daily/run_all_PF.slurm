#! /bin/bash

#SBATCH --partition=gpu

sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/PF/PoisFact_114_K25.slurm
