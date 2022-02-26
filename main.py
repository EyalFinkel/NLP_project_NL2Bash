import random
import wandb
import yaml
import os
import re
import openai
import argparse
from argparse import Namespace
from tqdm import tqdm
from box import Box
from typing import List, Union, Tuple
from time import time, sleep
from torch.utils.data import random_split, DataLoader

from metric.metric_utils import compute_metric
from dataset import NL2BashDataset
from utils import set_seed


def get_args() -> Union[Box, Namespace]:
    """
    Parses arguments from config file if given, else returns command line arguments.
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='', type=str)

    parser.add_argument('--entity', default='yuval_eyal', type=str)
    parser.add_argument('--project', default='nl2bash', type=str)
    parser.add_argument('--env_var', default='OPENAI_API_KEY', type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('-p', '--data_path', default='nl2bash-data.json', type=str)

    parser.add_argument('--engine', default='code-davinci-001', type=str)
    parser.add_argument('-n', '--num_val_samples', default=1000, type=int)
    parser.add_argument('--val_batch', default=5, type=int)
    parser.add_argument('--example_batch', default=3, type=int)
    parser.add_argument('-t', '--max_tokens', default=1500, type=int)
    parser.add_argument('--prompt_method', default=1, type=int)
    parser.add_argument('--omit_rate', default=0.0, type=float)
    args = parser.parse_args()

    if args.config != '':
        with open(args.config, "r") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        args = Box(args)
    else:
        delattr(args, 'config')
    return args


def make_prompt(invs: Tuple[str], example_invs: Tuple, example_gt_cmds: Tuple, prompt_method: int = 2) -> str:
    """
    Returns the prompt containing the example samples and evaluated samples, according to the given prompt structure.
    :param invs: invocations of the evaluated samples
    :param example_invs: invocations of the example samples
    :param example_gt_cmds: bash command ground truths of the example samples
    :param prompt_method: the structure of the prompt, excepts values in [1, 2, 3, 4]
                          1 - numbered examples intertwined with evaluated samples
                          2 - numbered examples separated from evaluated samples
                          3 - prompted examples intertwined with evaluated samples
                          4 - prompted examples separated from evaluated samples
    :return: the prompt for the model
    """
    num_examples = len(example_invs)
    prompt = "# Translate the following set of instructions to Bash command:\n"  # common prefix for all structures

    if prompt_method in [1, 3]:
        prompt = f"{prompt}# Invocations:\n"
        for i, inv in enumerate(example_invs + invs):
            prompt = f"{prompt}# {i + 1}. {inv}\n"
        prompt = f"{prompt}\n# Bash commands:\n"
        for i, example_gt_cmd in enumerate(example_gt_cmds):
            prompt = f"{prompt}{f'{i + 1}.' if prompt_method == 1 else 'bash>'} {example_gt_cmd}\n"

    elif prompt_method in [2, 4]:
        if num_examples > 0:
            prompt = f"{prompt}# Invocations:\n"
            for i, example_inv in enumerate(example_invs):
                prompt = f"{prompt}# {i + 1}. {example_inv}\n"
            prompt = f"{prompt}\n# Bash commands:\n"
            for i, example_gt_cmd in enumerate(example_gt_cmds):
                prompt = f"{prompt}{f'{i + 1}.' if prompt_method == 2 else 'bash>'} {example_gt_cmd}\n"
            prompt = f"{prompt}\n"
        prompt = f"{prompt}# Invocations:\n"
        for i, inv in enumerate(invs):
            prompt = f"{prompt}# {i + 1}. {inv}\n"
        prompt = f"{prompt}\n# Bash commands:"

    else:
        raise NotImplemented(f"Prompt method {prompt_method} is not supported.")

    return prompt


def random_omit(prompt: str, omit_rate: float = 0.1) -> str:
    """
    Omits random words from the prompt according to the given omit rate.
    :param prompt: the prompt for the model
    :param omit_rate: a rate to determine how many words to omit
    :return: the prompt with words omitted
    """
    # split prompt by spaces, keeping \n's with preceding word and removing empty strings
    prompt_split = [word for element in re.split(' +', prompt) for word in re.split('(\w*\n+)', element)]
    prompt_split = list(filter(None, prompt_split))

    num_omit = round(len(prompt_split) * omit_rate)
    for _ in range(num_omit):
        i = random.choice(range(len(prompt_split)))
        count = prompt_split[i].count('\n')
        # if there are '\n's following the omitted word, add them to the previous word to preserve the prompt structure
        if count > 0:
            prompt_split[i - 1] += count * '\n'
        prompt_split.pop(i)

    # reconstruct the new prompt
    new_prompt = ' '.join(prompt_split)
    return re.sub('\n ', '\n', new_prompt)


def get_predictions(engine: str, prompt: str, max_tokens: int, last_sleep: float) -> Tuple[str, float]:
    """
    Sends the prompt to the specified model and returns the response.
    :param engine: the model to infer on
    :param prompt: the prompt to send to the model
    :param max_tokens: max tokens for the request and the response combined
    :param last_sleep: last time at which a rate limit error received and the program went to sleep
    :return: the response from the model and the last sleep time
    """
    success = False
    # continue trying to send the request until it succeeds
    while not success:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                stop=['\n\n']
            )
            success = True
            response_str = response["choices"][0]["text"].strip('\n')  # get the response string
            return response_str, last_sleep

        # handle the case of rate limit exception
        except openai.error.RateLimitError:
            # sleep enough time if the rate limit exceeded to enable future requests
            sleep(65 - (time() - last_sleep) % 60)
            last_sleep = time()


def postprocess(response: str) -> List[str]:
    """
    Extracts the bash commands from the model's response.
    :param response: the response from the model
    :return: the bash command predictions
    """
    preds = response.split("\n")
    processed_preds = []

    # remove prefixes from the predictions, such as numbers (1., 2., ...), comments (#) and bash prompts (bash>)
    pattern = "^#*\s*(?:\d+\.|bash>)\s*"
    for pred in preds:
        processed_preds.append(re.sub(pattern, "", pred))

    return processed_preds


def main():
    """
    Runs the whole inference and evaluation process.
    :return:
    """
    args = get_args()
    set_seed(args.seed)
    openai.api_key = os.getenv(args.env_var)
    debug = args.debug
    entity = args.entity
    project = args.project
    data_path = args.data_path

    # initialize a wandb run
    if not debug:
        wandb.init(entity=entity, project=project, config=args, config_exclude_keys=["entity", "project", "debug", "data_path", "env_var"])
        args = wandb.config

    dataset = NL2BashDataset(data_path)  # create a dataset
    num_samples = len(dataset)
    # split data to evaluation set and a set dedicated for prompt examples
    val_set, example_set = random_split(dataset, [args.num_val_samples, num_samples - args.num_val_samples])
    # create dataloaders for both sets of data
    val_dataloader = DataLoader(val_set, batch_size=args.val_batch)
    example_dataloader = None if args.example_batch == 0 else DataLoader(example_set, batch_size=args.example_batch, shuffle=True)

    scores = []
    for i, (inv, gt_cmd) in enumerate(tqdm(val_dataloader)):
        # get a single batch of examples
        example_inv, example_gt_cmd = ((), ()) if example_dataloader is None else next(iter(example_dataloader))

        # build the prompt
        prompt = make_prompt(inv, example_inv, example_gt_cmd, prompt_method=args.prompt_method)
        if args.omit_rate > 0:
            prompt = random_omit(prompt, omit_rate=args.omit_rate)

        if i == 0:
            last_sleep = time()
        response, last_sleep = get_predictions(args.engine, prompt, args.max_tokens, last_sleep)  # infer the model
        preds = postprocess(response)  # process the model's response

        # evaluate predictions
        for j in range(min(args.val_batch, len(preds))):
            score = compute_metric(preds[j], 1.0, gt_cmd[j])
            scores.append(score)

    # calculate the score and log it to wandb
    tot_score = sum(scores) / len(scores)
    print("Total score:", tot_score)
    if not debug:
        wandb.log({"score": tot_score})


if __name__ == "__main__":
    main()
