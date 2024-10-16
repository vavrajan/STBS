import os

data_name = 'hein-daily'
addenda = range(114, 115)

### First set up directories on the cluster:
# project_dir = 'home/jvavra/STBS/'
project_dir = os.getcwd()
slurm_dir = os.path.join(project_dir, 'slurm', data_name)
stbs_dir = os.path.join(slurm_dir, 'STBS')
out_dir = os.path.join(project_dir, 'out', data_name)
err_dir = os.path.join(project_dir, 'err', data_name)
code_dir = os.path.join(project_dir, 'code')
analysis_dir = os.path.join(project_dir, 'analysis')

if not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
if not os.path.exists(stbs_dir):
    os.mkdir(stbs_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)

# For now just use the environment for testing.
partition = 'gpu'

# Different number of topics
Ks = [25]
num_epochs = 150

## Different model settings
scenarios = {}
# classical TBIP with Gamma variational families by Keyon and Vafa
for i in addenda:
    addendum = str(i)
    for K in Ks:
        name = "TBIP_" + addendum + "_K" + str(K)
        scenarios[name] = {"checkpoint_name": name,
                           "addendum": addendum,
                           "num_topics": K,
                           "computeIC_every": 5,
                           "pre_initialize_parameters": True,
                           "exact_entropy": False, "geom_approx": True,
                           "iota_coef_jointly": False, # irrelevant
                           "theta": "Gfix", "exp_verbosity": "LNfix", "beta": "Gfix",
                           "eta": "Nfix",
                           "covariates": "None", # irrelevant
                           "ideal_dim": "a", "ideal_mean": "Nfix", "ideal_prec": "Nfix",
                           "iota_dim": "l", "iota_mean": "None", "iota_prec": "Nfix"} # irrelevant
# ideological positions fixed for all topics
for i in addenda:
    addendum = str(i)
    for K in Ks:
        for covariates in ["party", "all_no_int", "all"]:
            name = "STBS_ideal_a_" + covariates + addendum + "_K" + str(K)
            scenarios[name] = {"checkpoint_name": name,
                               "addendum": addendum,
                               "num_topics": K,
                               "computeIC_every": 5,
                               "pre_initialize_parameters": True,
                               "exact_entropy": True, "geom_approx": False,
                               "iota_coef_jointly": True,
                               "theta": "Garte", "exp_verbosity": "None", "beta": "Gvrte",
                               "eta": "NkprecF",
                               "covariates": covariates,
                               "ideal_dim": "a", "ideal_mean": "Nreg", "ideal_prec": "Nprec",
                               "iota_dim": "l", "iota_mean": "None", "iota_prec": "NlprecG"}
# topic-specific ideological positions
for i in addenda:
    addendum = str(i)
    for K in Ks:
        for covariates in ["party", "all_no_int", "all"]:
            name = "STBS_ideal_ak_" + covariates + addendum + "_K" + str(K)
            scenarios[name] = {"checkpoint_name": name,
                               "addendum": addendum,
                               "num_topics": K,
                               "computeIC_every": 5,
                               "pre_initialize_parameters": True,
                               "exact_entropy": True, "geom_approx": False,
                               "iota_coef_jointly": True,
                               "theta": "Garte", "exp_verbosity": "None", "beta": "Gvrte",
                               "eta": "NkprecF",
                               "covariates": covariates,
                               "ideal_dim": "ak", "ideal_mean": "Nreg", "ideal_prec": "Naprec",
                               "iota_dim": "kl", "iota_mean": "Nlmean", "iota_prec": "NlprecF"}



### Creating slurm files and one file to trigger all the jobs.
with open(os.path.join(slurm_dir, 'run_all_STBS.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    all_file.write('#SBATCH --partition=' + partition + '\n')
    all_file.write('\n')
    for name in scenarios:
        flags = '  --num_epochs='+str(num_epochs) + '  --data_name='+data_name
        for key in scenarios[name]:
            flags = flags+'  --'+key+'='+str(scenarios[name][key])
        stbs_path = os.path.join(stbs_dir, name+'.slurm')
        with open(stbs_path, 'w') as file:
            file.write('#!/bin/bash\n')
            file.write('#SBATCH --job-name=STBS_' + data_name + ' # short name for your job\n')
            file.write('#SBATCH --partition='+partition+'\n')

            # Other potential computational settings.
            # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
            # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
            # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
            # file.write('#SBATCH --mem=1G # total memory per node\n')
            file.write('#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task\n')
            # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
            # file.write('#SBATCH --mail-type=ALL\n')

            file.write('#SBATCH -o ' + os.path.join(out_dir, '%x_%j_%N.out') + '      # save stdout to file. '
                                                                               'The filename is defined through filename pattern\n')
            file.write('#SBATCH -e ' + os.path.join(err_dir, '%x_%j_%N.err') + '      # save stderr to file. '
                                                                               'The filename is defined through filename pattern\n')
            file.write('#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)\n')
            file.write('\n')
            file.write('. /opt/apps/2023-04-11_lmod.bash\n')
            file.write('ml purge\n')
            file.write('ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn\n')
            file.write('\n')
            file.write('cd /home/jvavra/STBS/\n')
            file.write('conda activate tf_TBIP\n')
            file.write('\n')
            file.write('python '+os.path.join(analysis_dir, 'estimate_STBS_cluster.py')+flags+'\n')
        # Add a line for running the batch script to the overall slurm job.
        all_file.write('sbatch --dependency=singleton ' + stbs_path)
        all_file.write('\n')
