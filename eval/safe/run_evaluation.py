from search_augmented_factuality_eval import main
from common.modeling import Model
from langchain_openai import AzureChatOpenAI, AzureOpenAI
import json
import os
import argparse
import json
import multiprocessing

def append_to_jsonl_file(content, file_path):
    with open(file_path, 'a') as s:
        s.write(json.dumps(content))
        s.write("\n")

def get_data(input_path):

    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # topics = [o["topic"] for o in data]
    # generations = [o["output"] for o in data]

    return data

def evaluate(data, ok):
    question, answer = data["topic"], data["output"]
    # out = fs.get_score(topics, generations, atomic_facts=atomic_facts, gamma=10, knowledge_source="enwiki-20230401", verbose=True)
    out = main(question, answer)
    # out["experiment_name"] = experiment_name
    # print(f'Experiment {experiment_name} \n\t\tScore {out["score"]} \t init score {out["init_score"]} \t respons ratio {out["respond_ratio"]} \t facts per response {out["num_facts_per_response"]}')

    with open(f"eval_factscore.json", 'a') as outfile:
        outfile.write(json.dumps(out))

    return 0

if __name__ == '__main__':

    input_dir = './Cove.jsonl'

    all_data = get_data(input_dir)[:2]

    print(all_data)
    num_cpus = multiprocessing.cpu_count()
    if len(all_data) < multiprocessing.cpu_count():
        num_cpus = len(all_data)
    pool = multiprocessing.Pool(num_cpus)
    print(f"Will be using {num_cpus} cpus.")

    jobs = []
    for data in all_data:
        jobs.append(pool.apply_async(evaluate, (data, "o")))

    results = [job.get() for job in jobs]
    print(results)

    print("Experiment(s) done.")