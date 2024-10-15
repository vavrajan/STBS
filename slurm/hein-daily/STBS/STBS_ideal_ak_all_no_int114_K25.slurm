#!/bin/bash
#SBATCH --job-name=STBS_hein-daily # short name for your job
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task
#SBATCH -o /home/jvavra/STBS/out/hein-daily/%x_%j_%N.out      # save stdout to file. The filename is defined through filename pattern
#SBATCH -e /home/jvavra/STBS/err/hein-daily/%x_%j_%N.err      # save stderr to file. The filename is defined through filename pattern
#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)

. /opt/apps/2023-04-11_lmod.bash
ml purge
ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn

cd /home/jvavra/STBS/
conda activate tf_TBIP

python /home/jvavra/STBS/analysis/estimate_STBS_cluster.py  --num_epochs=1000  --data_name=hein-daily  --checkpoint_name=STBS_ideal_ak_all_no_int114_K25  --addendum=114  --num_topics=25  --pre_initialize_parameters=True  --exact_entropy=True  --geom_approx=False  --iota_coef_jointly=True  --theta=Garte  --exp_verbosity=None  --beta=Gvrte  --eta=NkprecF  --covariates=all_no_int  --ideal_dim=ak  --ideal_mean=Nreg  --ideal_prec=Naprec  --iota_dim=kl  --iota_mean=Nlmean  --iota_prec=NlprecF
