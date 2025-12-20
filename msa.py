from openai import OpenAI
import os
import json

os.environ['OPENAI_API_KEY'] = "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"

client = OpenAI()

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    s = ''
    for l in lines:
        s += l
    return s


# Hyperparameters, Filenames

id = 4

input_filename = f'scenarios/gpt-5p1-{id}.txt'
output_program_filename = f'programs/pg-gpt-5p1-{id}.wppl'
output_json_filename = f'inference_results/result-10000-gpt-5p1-{id}.json'

# input_filename = 'pg-scenario.txt'
# output_program_filename = 'pg.wppl'
# output_json_filename = 'result-existing-canoe-10000.json'

temperature = 0.0

scenario = read_file(input_filename)
# print(scenario)
# print('=================================')

# PART I - Parse

part1prompt = read_file('pg-part1.txt')
part1prompt_program = part1prompt + '\n\n' + scenario

response = client.responses.create(
    model="gpt-4o",
    temperature=temperature,
    input=part1prompt_program
)

current_program = scenario + '\n\n' + response.output_text

# PART II - Knowledge

part2prompt = read_file('pg-part2.txt')
part2prompt_program = part2prompt + '\n\n' + current_program

response = client.responses.create(
    model="gpt-4o",
    temperature=temperature,
    input=part2prompt_program
)

tmp_program1 = current_program
current_program = current_program + '\n\n' + response.output_text

# PART III - Write Model

part3prompt = read_file('pg-part3.txt')
part3prompt_program = part3prompt + '\n\n' + current_program

response = client.responses.create(
    model="gpt-4o",
    temperature=temperature,
    input=part3prompt_program
)

tmp_program2 = current_program
current_program = current_program + '\n\n' + response.output_text
webppl_program = response.output_text.split('<START_WEBPPL_MODEL>\n')[1].split('\n<END_WEBPPL_MODEL>')[0]
# webppl_program += f'\njson.write(\'{output_json_filename}\', posterior);'
additional_helpers = read_file('additional_helpers.txt')
webppl_program = additional_helpers + '\n' + webppl_program

print(webppl_program)
with open(output_program_filename, "w") as f:
    f.write(webppl_program)
