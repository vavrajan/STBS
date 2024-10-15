#! /bin/bash

#SBATCH --partition=gpu

sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/TBIP_114_K25.slurm
sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/STBS_ideal_a_party114_K25.slurm
sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/STBS_ideal_a_all_no_int114_K25.slurm
sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/STBS_ideal_a_all114_K25.slurm
sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/STBS_ideal_ak_party114_K25.slurm
sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/STBS_ideal_ak_all_no_int114_K25.slurm
sbatch --dependency=singleton /home/jvavra/STBS/slurm/hein-daily/STBS/STBS_ideal_ak_all114_K25.slurm
