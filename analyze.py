import sys
import json
import string
import re

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from control_codes import START_C_CODES


CONST = {
    'FormatEmail': "E-mails",
    'FormatForum': "Forum messages",
    'FormatMono': "Monologue",
    'FormatQA': "Q&A",
    'FormatOther': "Other",
    'SemJunk': 'Incomprehensible',
    'BadFlow': 'Incoherent',
    'VisBadFlow': 'Is incoherent?',
    'Junk': "Junk text",
    'RepText': "Repetitive chunks",
    'VisRepText': "Has repetitive chunks?",
    'ContMism': "Initial continuation mismatch",
    'VisMatchPrompt': "The first utterance\nmatches the prompt?",
    "FactErrors": 'Has factual errors?',
    "GrammarErrors": "Has grammatical errors?"
}

MODEL_DICT = {
    "sample_occ": "Sampling (OCC)",
    "greedy_occ": "Greedy (OCC)",
    "sample_prompt": "Sampling (prompt)",
    "greedy_prompt": "Greedy (prompt)"
}

TEXT_TYPE_DICT = {
    'non-conversational': 'non-CONV',
    'partially conversational': 'partly CONV',
    'conversational': 'CONV'
}

def minify(x):
    x = re.sub("[{}]".format(string.punctuation), "", x)
    x = re.sub("[0-9]", "", x)
    x = x.replace("\n", "").replace(" ", "")
    x = x.replace("\ufeff", "")
    return x.strip()


if __name__ == '__main__':
    source_files = {
        'data': ['swectrl_conv_data.json'],
        'keys': ['swectrl_conv_key.json']
    }

    eval_file = "annotations.json"

    with open("prompts.yaml") as f:
        prompts = yaml.load(f, yaml.Loader)
        p2c = {
            p: h
            for h, v in prompts.items()
            for p in v
        }
    
    df_data = []
    sf = source_files
    ef = eval_file
    
    orig_data = []
    for fn in sf['data']:
        with open(fn) as f:
            orig_data.extend(json.load(f))
    
    with open(ef) as f:
        res_data = sorted(json.load(f)['data'], key=lambda x: int(x['num']))

    assert (len(orig_data) == len(res_data)), "Different sizes of original and annotated data!\n{} original vs {} annotated".format(
        len(orig_data), len(res_data)
    )

    for a, b in zip(orig_data, res_data):
        assert (minify(a['text'])[:70] == minify(b['context'])[:70]), "Found unequal datapoints:\n{}\n{}".format(a, b)
    print("ALL GOOD!")

    key = []
    for fn in sf['keys']:
        with open(fn) as f:
            key.extend(json.load(f))
    
    df_data = []
    for r_data, key_data in zip(res_data, key):
        model, vcat = key_data
        cat = vcat.replace("_", "/")

        problems, text_type = [], None
        extra_utt, num_parties = None, None
        shifted_format = []
        has_factual_errors, has_grammar_errors = False, False
        for ann in r_data['annotations']:
            if ann.get('labels'):
                for lab in ann['labels']:
                    if lab['marker']['name'] == 'Factual error':
                        has_factual_errors = True
                    elif lab['marker']['name'] == 'Grammatical error':
                        has_grammar_errors = True

            if ann.get('inputs'):
                for inp in ann['inputs']:
                    if inp['marker']['name'] == 'Problems':
                        problems = inp['content'].split("||")
                    elif inp['marker']['name'] == 'Type of text':
                        text_type = inp['content'].strip()
                    elif inp['marker']['name'] == 'Number of parties':
                        num_parties = inp['content'].strip()
                    elif inp['marker']['name'] == 'Extra utterances':
                        extra_utt = inp['content'].strip()
                    elif inp['marker']['name'] == 'Shifted format':
                        shifted_format = inp['content'].split("||")
        dp = {
            "model": MODEL_DICT[model],
            "Category": cat,
            "num": r_data['num']
        }
        
        if text_type is not None:
            dp["text_type"] = TEXT_TYPE_DICT[text_type]

        if num_parties is not None:
            dp["num_parties"] = num_parties

        if extra_utt is not None:
            dp["xutt"] = extra_utt
        
        for p in problems:
            dp[p] = True
        for sf in shifted_format:
            dp[sf] = True
        dp['fact_errors'] = has_factual_errors
        dp['grammar_errors'] = has_grammar_errors
        df_data.append(dp)
 
    df = pd.DataFrame.from_dict(df_data)
    df['Model'] = pd.Categorical(
        df['model'], categories=MODEL_DICT.values()
    )
    df['Text type'] = pd.Categorical(
        df['text_type'], categories=TEXT_TYPE_DICT.values()
    )
    print(df.shape)
    print(df.head())

    # Seaborn settings
    sns.set_theme(style="dark", font_scale=1.5)
    
    g = sns.displot(
        kind='hist',
        data=df,
        x='Text type',
        hue='Model',
        col='Category',
        col_wrap=4,
        discrete=True,
        multiple='stack'
    )
    for ax in g.axes:
        ax.grid(axis='y')
    plt.savefig("text_types.pdf", bbox_inches='tight')
    
    with pd.option_context('mode.chained_assignment', None):
        conv_df = df[df['text_type'] != 'non-CONV']
        conv_df[CONST['RepText']].fillna(False, inplace=True)
        conv_df[CONST['VisRepText']] = pd.Categorical(
            conv_df[CONST['RepText']].apply(lambda x: 'Yes' if x else 'No'),
            categories=('Yes', 'No')
        )
        print("CONV", conv_df.shape)
        g = sns.displot(
            kind='hist',
            data=conv_df,
            x=CONST['VisRepText'],
            hue='Model',
            col='Category',
            col_wrap=4,
            discrete=True,
            multiple='stack'
        ) 
        for ax in g.axes:
            ax.grid(axis='y')

        no_rep_df = conv_df[conv_df[CONST['VisRepText']] == 'No']
        print("NO REP", no_rep_df.shape)
        no_rep_df[CONST['ContMism']].fillna(False, inplace=True)
        no_rep_df[CONST['VisMatchPrompt']] = pd.Categorical(
            no_rep_df[CONST['ContMism']].apply(lambda x: 'No' if x else 'Yes'),
            categories=('Yes', 'No')
        )
        g = sns.displot(
            kind='hist',
            data=no_rep_df,
            x=CONST['VisMatchPrompt'],
            hue='Model',
            col='Category',
            col_wrap=4,
            discrete=True,
            multiple='stack'
        ) 
        for ax in g.axes:
            ax.grid(axis='y')
        plt.savefig("prompt_match.pdf", bbox_inches='tight')

        match_df = no_rep_df[no_rep_df[CONST['VisMatchPrompt']] == 'Yes']
        print("MATCH PROMPT", match_df.shape)
        match_df[CONST['BadFlow']].fillna(False, inplace=True)
        match_df[CONST['VisBadFlow']] = pd.Categorical(
            match_df[CONST['BadFlow']].apply(lambda x: 'Yes' if x else 'No'),
            categories=('Yes', 'No')
        )
        g = sns.displot(
            kind='hist',
            data=match_df,
            x=CONST['VisBadFlow'],
            hue='Model',
            col='Category',
            col_wrap=4,
            discrete=True,
            multiple='stack'
        ) 
        for ax in g.axes:
            ax.grid(axis='y')
        plt.savefig("good_flow.pdf", bbox_inches='tight')
        
        good_flow_df = match_df[match_df[CONST['VisBadFlow']] == 'No']
        print("COHERENT", good_flow_df.shape)
        good_flow_df[CONST['FactErrors']] = pd.Categorical(
            good_flow_df['fact_errors'].apply(lambda x: 'Yes' if x else 'No'),
            categories=('Yes', 'No')
        )
        g = sns.displot(
            kind='hist',
            data=good_flow_df,
            x=CONST['FactErrors'],
            hue='Model',
            col='Category',
            col_wrap=4,
            discrete=True,
            multiple='stack'
        ) 
        for ax in g.axes:
            ax.grid(axis='y')

        plt.savefig("factual_errors.pdf", bbox_inches='tight')

        no_fact_errors_df = good_flow_df[good_flow_df[CONST['FactErrors']] == 'No']
        print("NO FACT ERRORS", no_fact_errors_df.shape)
        no_fact_errors_df[CONST['GrammarErrors']] = pd.Categorical(
            no_fact_errors_df['grammar_errors'].apply(lambda x: 'Yes' if x else 'No'),
            categories=('Yes', 'No')
        )
        g = sns.displot(
            kind='hist',
            data=no_fact_errors_df,
            x=CONST['GrammarErrors'],
            hue='Model',
            col='Category',
            col_wrap=4,
            discrete=True,
            multiple='stack'
        ) 
        for ax in g.axes:
            ax.grid(axis='y')
        plt.savefig("grammar_errors.pdf", bbox_inches='tight')
        print(no_fact_errors_df[no_fact_errors_df[CONST['GrammarErrors']] == 'No'])
        print(",".join(map(str, no_fact_errors_df[no_fact_errors_df[CONST['GrammarErrors']] == 'No']['num'].tolist())))
        print(",".join(map(str, no_fact_errors_df[no_fact_errors_df[CONST['GrammarErrors']] == 'Yes']['num'].tolist())))

