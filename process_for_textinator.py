import os
import sys
import re
import json
import glob
import random
import argparse
from collections import defaultdict
from pathlib import Path

import jsonlines as jsl

from control_codes import START_C_CODES, END_C_CODES


def remove_cc(text):
    start_c_codes = sorted(START_C_CODES.values(), key=lambda x: x.find("::"), reverse=True)
    end_c_codes = sorted(END_C_CODES.values(), key=lambda x: x.find("::"), reverse=True)
    
    for codes in (end_c_codes, start_c_codes):
        for x in codes:
            text = text.replace(x, "")

    return re.sub(" {2,}", " ", text).strip()

CAT = {
    "sp": "sample_prompt",
    "gp": "greedy_prompt",
    "so": "sample_occ",
    "go": "greedy_occ"
}

FILE2MODEL = {
    "debate_1683457252.jsonl": CAT["sp"],
    'debate_1683457451.jsonl': CAT["gp"],
    "debate_1683518681.jsonl": CAT["so"],
    "debate_1683518780.jsonl": CAT["go"],
    "forum_1683402685.jsonl": CAT["sp"],
    "forum_1683403101.jsonl": CAT["gp"],
    "forum_1683516438.jsonl": CAT["so"],
    "forum_1683516539.jsonl": CAT["go"],
    "forum_economy_1683411016.jsonl": CAT["sp"],
    "forum_economy_1683411403.jsonl": CAT["gp"],
    "forum_economy_1683517967.jsonl": CAT["so"],
    "forum_economy_1683518067.jsonl": CAT["go"],
    "forum_law_1683451520.jsonl": CAT["sp"],
    "forum_law_1683452008.jsonl": CAT["gp"],
    "forum_law_1683518407.jsonl": CAT["so"],
    "forum_law_1683518507.jsonl": CAT["go"],
    "forum_sport_1683485998.jsonl": CAT["sp"],
    "forum_sport_1683486499.jsonl": CAT["gp"],
    "forum_sport_1683518925.jsonl": CAT["so"],
    "forum_sport_1683519006.jsonl": CAT["go"],
    "forum_tech_1683487086.jsonl": CAT["sp"],
    "forum_tech_1683487498.jsonl": CAT["gp"],
    "forum_tech_1683516344.jsonl": CAT["so"],
    "forum_tech_1683516444.jsonl": CAT["go"],
    "forum_travel_1683452313.jsonl": CAT["sp"],
    "forum_travel_1683452802.jsonl": CAT["gp"],
    "forum_travel_1683517177.jsonl": CAT["so"],
    "forum_travel_1683517279.jsonl": CAT["go"],
    "news_1683453122.jsonl": CAT["sp"],
    "news_1683453318.jsonl": CAT["gp"],
    "news_1683517530.jsonl": CAT["so"],
    "news_1683517630.jsonl": CAT["go"]
}


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))

    eval_files = glob.glob(os.path.join(base_dir, "*.jsonl"))

    data = defaultdict(lambda: defaultdict(list))

    for fname in eval_files:
        with jsl.open(fname) as f:
            fname_proper = Path(fname).name
            cat = fname_proper.rpartition("_")[0]
            for obj in f:
                prompt = obj["prompt"]
                text = obj["text"]
                data[cat][prompt].append((
                    FILE2MODEL[fname_proper],
                    cat,
                    prompt,
                    remove_cc(text)
                ))

    tt_data = []
    tt_models = []

    cats = list(data.keys())
    N_cats = len(cats)

    for i, cat in enumerate(cats):
        for prompt, gens in data[cat].items():
            random.shuffle(gens)
            
            for model, vcat, vprompt, gen in gens:
                tt_data.append({
                    "text": "{}\n!-^-!\n__Task:__ {}".format(gen, vprompt)
                })
                tt_models.append((model, vcat))

    with open(os.path.join(base_dir, "swectrl_conv_data.json".format(i)), 'w') as f:
        json.dump(tt_data, f)
    with open(os.path.join(base_dir, "swectrl_conv_key.json".format(i)), 'w') as f:
        json.dump(tt_models, f)


