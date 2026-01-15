
import subprocess
import time
import json
import datetime
import numpy as np

tm = str(datetime.datetime.now())
TMSTR = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:19]
program_ids = []
for P in range(4):
    for C in range(2):
        for R in range(2):
            for N in range(2,8):
                for seed in range(5):
                    program_ids.append(f'P-{P}-C-{C}-R-{R}-N-{N}-{seed}')

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    s = ''
    for l in lines:
        s += l
    return s

def write_run(i, program_id='', tmstr=''):
    with open(f'temp-{program_id}-{tmstr}.sh', 'w') as f:
        f.write(f"#!/bin/bash\n"
                "#SBATCH --job-name={0}\n"
                "#SBATCH --time=23:59:59\n"
                #"#SBATCH --gres=gpu:1\n"
                #"#SBATCH --partition=mig\n"
                # "#SBATCH --partition=pli\n"
                # "#SBATCH --account=bayesllm\n"
                # "#SBATCH --constraint=gpu80\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem-per-cpu=50G\n"
                "#SBATCH --mail-type=begin\n"
                "#SBATCH --mail-type=end\n"
                "#SBATCH --mail-user=zhangliyi97@gmail.com\n".format(i)
                )
        cmd = f"webppl programs/tmp/pg-{program_id}-{tmstr}.wppl --require webppl-json"
        f.write(cmd+'\n')

    subprocess.call(f'chmod +x temp-{program_id}-{tmstr}.sh', shell=True)
    time.sleep(0.1)
    subprocess.call(f'sbatch temp-{program_id}-{tmstr}.sh', shell=True)


for program_id in program_ids:
    TMSTR_LIST = []
    for i in range(20):
        # save = '_'.join( [x.strip().split(' ')[-1] for x in ar.split('--') if len(x.strip()) > 0] )
        print(i)
        TMSTR_LIST.append(TMSTR + '-' + str(i))
        TMSTR_i = TMSTR_LIST[i]
        result_json_filename = f'inference_results/result-rejection-{program_id}-{TMSTR_i}.json'
        samples_json_filename = f'inference_results/samples-rejection-{program_id}-{TMSTR_i}.json'
        program_filename = f'programs/pg-{program_id}.wppl'

        # Create a copy of the webppl program, with an addtional line specifying save filename
        webppl_program = read_file(program_filename)
        webppl_program += f'\njson.write(\'{result_json_filename}\', posterior);'
        webppl_program += f'\njson.write(\'{samples_json_filename}\', posterior.samples);'

        with open(f"programs/tmp/pg-{program_id}-{TMSTR_i}.wppl", "w") as f:
            f.write(webppl_program)

    for i in range(20):
        # Run this inference program
        print(i)
        write_run(str(i), program_id, TMSTR_LIST[i])

        # # Read inference results
        # with open(result_json_filename, 'r') as f:
        #     data = json.load(f)

        # results = {}
        # for key in data['support'][0]:
        #     results[key] = {}

        # for i in range(len(data['probs'])):
        #     for query in results:
        #         if data['support'][i][query] not in results[query]:
        #             results[query][data['support'][i][query]] = data['probs'][i]
        #         else:
        #             results[query][data['support'][i][query]] += data['probs'][i]

        # sum_results = {key:0 for key in results}
        # for key in results:
        #     for estimate in results[key]:
        #         sum_results[key] += estimate * results[key][estimate]

        # print(sum_results)
