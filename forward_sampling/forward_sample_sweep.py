import subprocess
import time
import json
import argparse
import datetime

tm = str(datetime.datetime.now())
TMSTR = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]

def write_run(jobname, extra=''):
    with open('temp.sh', 'w') as f:
        f.write("#!/bin/bash\n"
                "#SBATCH --job-name={0}\n"
                "#SBATCH --time=10:00:00\n"
                #"#SBATCH --gres=gpu:1\n"
                #"#SBATCH --partition=mig\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem-per-cpu=20G\n"
                "#SBATCH --mail-type=begin\n"
                "#SBATCH --mail-type=end\n"
                "#SBATCH --mail-user=anonymous@example.com

        f.write('module load proxy/default \n')
        cmd = "python forward_sample.py "
        cat = " >archive/" + TMSTR + "_" + jobname + ".out"
        f.write(cmd+extra+cat+'\n')

    subprocess.call('chmod +x temp.sh', shell=True)
    time.sleep(0.1)
    subprocess.call('sbatch temp.sh', shell=True)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sweep', type=str, default='forward_sample_sweep.json',
                   help='sweep file')
args = parser.parse_args()
print(vars(args))

with open(args.sweep, 'r') as f:
    sweep_args = json.load(f)

arglist = ['']
for opt, vals in sweep_args.items():
    new_arglist = []
    for j, v in enumerate(vals):
        for i in range(len(arglist)):
            new_arglist.append( arglist[i] + ' --'+opt+' '+str(v) )
    arglist = new_arglist

for i, ar in enumerate(arglist):
    # save = '_'.join( [x.strip().split(' ')[-1] for x in ar.split('--') if len(x.strip()) > 0] )

    if i in [9]:
        print(i, ar)
        write_run(str(i), ar)
