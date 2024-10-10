import os

data_name = 'hein-daily'
addenda = range(114, 115)

### First set up directories on the cluster:
# project_dir = 'home/jvavra/STBS/'
project_dir = os.getcwd()
slurm_dir = os.path.join(project_dir, 'slurm', data_name)
pf_dir = os.path.join(slurm_dir, 'PF')
out_dir = os.path.join(project_dir, 'out', data_name)
err_dir = os.path.join(project_dir, 'err', data_name)
code_dir = os.path.join(project_dir, 'code')

if not os.path.exists(slurm_dir):
    os.mkdir(slurm_dir)
if not os.path.exists(pf_dir):
    os.mkdir(pf_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(err_dir):
    os.mkdir(err_dir)

# For now just use the environment for testing.
partition = 'gpu'

# Different number of topics
Ks = [25]

### Creating slurm files and one file to trigger all the jobs.
with open(os.path.join(slurm_dir, 'run_all_PF.slurm'), 'w') as all_file:
    all_file.write('#! /bin/bash\n\n')
    all_file.write('#SBATCH --partition=' + partition + '\n')
    all_file.write('\n')
    for i in addenda:
        addendum = str(i)
        for K in Ks:
            flags = '  --data='+data_name+'  --num_topics='+str(K)+'  --addendum='+addendum
            pf_path = os.path.join(pf_dir, 'PoisFact_'+addendum+"_K"+str(K)+'.slurm')
            with open(pf_path, 'w') as file:
                file.write('#!/bin/bash\n')
                file.write('#SBATCH --job-name=PF_' + data_name + ' # short name for your job\n')
                file.write('#SBATCH --partition='+partition+'\n')

                # Other potential computational settings.
                # file.write('#SBATCH --nodes=1 # number of nodes you want to use\n')
                # file.write('#SBATCH --ntasks=2 # total number of tasks across all nodes\n')
                # file.write('#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)\n')
                # file.write('#SBATCH --mem=1G # total memory per node\n')
                file.write('#SBATCH --mem-per-cpu=256000 # in megabytes, default is 4GB per task\n')
                # file.write('#SBATCH --mail-user=jan.vavra@wu.ac.at\n')
                # file.write('#SBATCH --mail-type=ALL\n')

                file.write('#SBATCH -o '+os.path.join(out_dir, '%x_%j_%N.out') + '      # save stdout to file. '
                                                                                 'The filename is defined through filename pattern\n')
                file.write('#SBATCH -e '+os.path.join(err_dir, '%x_%j_%N.err') + '      # save stderr to file. '
                                                                                 'The filename is defined through filename pattern\n')
                file.write('#SBATCH --time=11:59:59 # total run time limit (HH:MM:SS)\n')
                file.write('\n')
                file.write('. /opt/apps/2023-04-11_lmod.bash\n')
                file.write('ml purge\n')
                file.write('ml miniconda3-4.10.3-gcc-12.2.0-ibprkvn\n')
                file.write('\n')
                file.write('cd /home/jvavra/STBS/\n')
                file.write('conda activate env_TBIP\n')
                file.write('\n')
                file.write('python '+os.path.join(code_dir, 'poisson_factorization.py')+flags+'\n')
            # Add a line for running the batch script to the overall slurm job.
            all_file.write('sbatch --dependency=singleton ' + pf_path)
            all_file.write('\n')