import os
import logging
import subprocess
import datetime
import sys
import time

from hydra import utils
from pathlib import Path
from omegaconf import OmegaConf, listconfig

user = os.environ['USER']
log = logging.getLogger(__name__)

hdd = "/scratch/hdd001/home/" + user
ssd = '/scratch/ssd001/home/' + user

date = datetime.datetime.now().strftime("%Y-%m-%d")

def eval_val(val):
    if 'eval:' in str(val):
        return str(eval(val.split('eval:', 1)[1]))
    else:
        return str(val)

def resolve_name(name):
    if isinstance(name, listconfig.ListConfig):
        name_list = [eval_val(str(name[i])) for i in range(len(name)) if name[i] != None]
        return '_'.join(name_list)
    else:
        return eval_val(name)

def get_j_dir(cfg):
    return os.path.join(ssd, "slurm", date, resolve_name(cfg.slurm.job_name))

def get_data_dir(cfg):
    return os.path.join('/scratch', 'ssd001', 'datasets', 'cfg.data.task', 'cfg.data.name')

def write_slurm(cfg):
    # set up run directories
    j_dir = get_j_dir(cfg)

    scripts_dir = os.path.join(j_dir, "scripts")
    if not os.path.exists(scripts_dir):
        Path(scripts_dir).mkdir(parents=True, exist_ok=True)

    slurm_opts = ['#SBATCH --' + k.replace('_','-') + '=' + resolve_name(v) for k, v in cfg.slurm.items() if v != None]

    # default output and error directories
    if not hasattr(cfg.slurm, 'output'):
        slurm_opts.append('#SBATCH --output={}/log/%j.out'.format(j_dir))
    if not hasattr(cfg.slurm, 'error'):
        slurm_opts.append('#SBATCH --error={}/log/%j.err'.format(j_dir))

    sh_path = os.path.join(j_dir, "scripts", resolve_name(cfg.slurm.job_name) + '.sh')
    binds = list(cfg.singularity.binds)
    binds.append(f'{sh_path}:/root/script.sh')
    slurm_opts = ['#!/bin/bash'] \
                + slurm_opts \
                + ['mkdir ' + cfg.slurm_additional.checkpoint_dir] \
                + [cfg.singularity.bin_path + ' --debug -vvv exec --userns --nv --writable ' + ' --bind ' + ','.join(binds) + ' ' + cfg.singularity.sbox_path + ' chmod u+x /root/script.sh'] \
                + [cfg.singularity.bin_path + ' --debug -vvv exec --userns --nv --writable ' + ' --bind ' + ','.join(binds) + ' ' + cfg.singularity.sbox_path + ' /root/script.sh']

    # write slurm file
    with open(os.path.join(j_dir, "scripts", resolve_name(cfg.slurm.job_name) + '.slrm'), 'w') as slrmf:
        slrmf.write('\n'.join(slurm_opts))

def write_sh(cfg, overrides):

    # set up run directories
    j_dir = get_j_dir(cfg)
    hydra_cwd = os.getcwd()
    curr_cwd = utils.get_original_cwd()
    exec_path = os.path.join(curr_cwd, sys.argv[0])

    scripts_dir = os.path.join(j_dir, "scripts")
    if not os.path.exists(scripts_dir):
        Path(scripts_dir).mkdir(parents=True, exist_ok=True)

    if 'venv' in cfg.slurm:
        venv_sh = '. /h/$USER/venv/{}/bin/activate'.format(cfg.slurm.venv)
    else:
        venv_sh = ''

    with open(os.path.join(j_dir, "scripts", resolve_name(cfg.slurm.job_name) + '.sh'), 'w') as shf:
        shf.write(
"""#!/bin/bash
ln -s /checkpoint/$USER/$SLURM_JOB_ID {0}/$SLURM_JOB_ID
touch {0}/$SLURM_JOB_ID/DELAYPURGE
{2}
export WANDB_DIR={3}
{4} {5} {6} {7}
""".format(
            j_dir,
            hydra_cwd,
            venv_sh,
            cfg.slurm_additional.wandb_dir,
            cfg.slurm_additional.python_bin,
            cfg.slurm_additional.python_optstr,
            cfg.exec_path,
            overrides
        ))

def symlink_hydra(cfg, cwd):
    hydra_dir = os.path.join(get_j_dir(cfg), 'conf')
    if not os.path.exists(os.path.join(hydra_dir, os.environ['SLURM_JOB_ID'])):
        log.info('Symlinking {} : {}'.format(cwd, hydra_dir))
        if not os.path.exists(hydra_dir):
            Path(hydra_dir).mkdir(parents=True, exist_ok=True)
        os.symlink(cwd, os.path.join(hydra_dir, os.environ['SLURM_JOB_ID']), target_is_directory=True)

def launch_job(cfg):

    # set up run directories
    j_dir = get_j_dir(cfg)
    log_dir = os.path.join(j_dir, "log")
    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # launch safe only when < 100 jobs running
    num_running = int(subprocess.run('squeue -u $USER | grep R | wc -l', shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
    while(num_running > 100):
        log.info("{} jobs running, waiting to run more...".format(num_running))
        time.sleep(10)
        num_running = int(subprocess.run('squeue -u $USER | grep R | wc -l', shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8'))
    subprocess.run('sbatch {0}/scripts/{1}.slrm'.format(j_dir, resolve_name(cfg.slurm.job_name)), shell=True)
