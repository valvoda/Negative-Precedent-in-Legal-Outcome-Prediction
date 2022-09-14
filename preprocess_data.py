from tqdm import tqdm
import os
import json
import re
import pickle
from pathlib import Path
import numpy as np

import torch
from transformers import LongformerTokenizer, BertTokenizer, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def fix_claims(all_facts, all_claims, all_outcomes, case_id, all_arguments):

    # flat_outcomes = []
    # for c in all_outcomes:
    #     flat_outcomes += c
    # allowed = list(set(flat_outcomes))

    allowed = ['10', '11', '13', '14', '2', '3', '5', '6', '7', '8', '9', 'P1-1', 'P1-3', 'P4-2']

    new_claims, new_outcomes, new_ids, new_facts, new_arguments = [], [], [], [], []
    for claim, outcome, i, c_id, fact, argument in zip(all_claims, all_outcomes, range(len(all_claims)), case_id, all_facts, all_arguments):
        n_c = []
        for c in claim:
            if c in allowed:
                n_c.append(c)

        cnt = 0
        flag = True
        if len(n_c) > 0 and len(n_c) >= len(outcome):
            for x in outcome:
                if x not in n_c:
                    flag = False
            if flag:
                n_c.sort()
                outcome.sort()
                new_claims.append(n_c)
                new_outcomes.append(outcome)
                new_ids.append(c_id)
                new_facts.append(fact)
                new_arguments.append(argument)
            else:
                cnt += 1

    return new_facts, new_claims, new_outcomes, new_ids, new_arguments

def get_arguments(data):
    try:
        arguments = data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0]
        arguments = arguments.split("\n")
        arguments = [a.strip() for a in arguments]
        arguments = list(filter(None, arguments))
    except:
        return []
    return arguments

def get_data(pretokenized_dir, tokenizer, max_len):
    dataset_facts = []
    dataset_arguments = []
    dataset_claims = []
    dataset_outcomes = []
    dataset_ids = []

    paths = ['train', 'dev', 'test']
    out_path = ['train_augmented', 'dev_augmented', 'test_augmented']

    for case_path, out in zip(paths, out_path):

        all_facts = []
        all_arguments = []
        all_claims = []
        all_outcomes = []
        all_ids = []

        for item in tqdm(os.listdir("ECHR/precedent/"+case_path)):
            if item.endswith('.json'):
                with open(os.path.join("ECHR/precedent/"+case_path, item), "r") as json_file:
                    data = json.load(json_file)
                    try:
                        alleged_arguments = data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0].lower()
                        # claims = list(set(re.findall("article\s(\d{1,2})\s", alleged_arguments)))
                        convention_claims = list(set(re.findall("article\s(\d{1,2})\s.{0,15}convention", alleged_arguments)))
                    #     claims = [c for c in other_claims if int(c) <= 18]
                    except:
                        convention_claims = []

                    try:
                        alleged_arguments = \
                        data["text"].split("THE LAW")[1].split("FOR THESE REASONS, THE COURT UNANIMOUSLY")[0].lower()
                        # claims = list(set(re.findall("article\s(\d{1,2})\s", alleged_arguments)))
                        protocol_claims = list(
                            set(re.findall("article\s(\d{1,2})\s.{0,15}protocol.{0,15}(\d)", alleged_arguments)))
                        protocol_claims = ['P' + p[1] + "-" + p[0] for p in protocol_claims]
                    except:
                        protocol_claims = []

                    argument = get_arguments(data)
                    claims = list(set(convention_claims + protocol_claims))
                    data['claim'] = claims
                    data['arguments'] = argument

                with open(os.path.join('ECHR/precedent/'+out, item), "w") as out_file:
                    json.dump(data, out_file, indent=1)

                if len(claims) > 0:
                    all_facts.append(data["facts"])
                    all_claims.append(claims)
                    all_arguments.append(argument)
                    # print("claims", claims)
                    # print("outcomes", data["violated_articles"])
                    all_outcomes.append(data["violated_articles"])
                    case_id = str(data["case_no"])
                    all_ids.append(case_id)

        all_facts, all_claims, all_outcomes, all_ids, all_arguments = fix_claims(all_facts, all_claims, all_outcomes, all_ids, all_arguments)
        print(pretokenized_dir, len(all_facts))

        dataset_facts += all_facts
        dataset_claims += all_claims
        dataset_outcomes += all_outcomes
        dataset_arguments += all_arguments
        dataset_ids += all_ids

    split_dataset(pretokenized_dir, tokenizer, max_len, dataset_ids, dataset_facts, dataset_arguments, dataset_claims, dataset_outcomes)

def get_stats(data):
    data = np.array(data)
    stats = np.array([0 for i in range(len(data[0]))])
    cnt = 0
    for d in data:
        stats = stats + d
        if d.sum() > 0:
            cnt += 1

    return stats, cnt

def get_neg(claim_data, out_data):
    cdata = np.array(claim_data)
    odata = np.array(out_data)
    c_stats, c_cnt = get_stats(claim_data)
    out_stats, out_cnt = get_stats(out_data)
    stats = c_stats - out_stats
    cnt = 0
    for c, o in zip(cdata, odata):
        n = c - o
        if n.sum() > 0:
            cnt += 1

    return stats, cnt

def data_stats(claims, outcomes, type):
    c_stats, c_cnt = get_stats(claims)
    out_stats, out_cnt = get_stats(outcomes)
    neg_stats = c_stats - out_stats
    _, n_cnt = get_neg(claims, outcomes)

    print("-" * 40)
    print(
        f"{type:^9} | {c_cnt:^9} | {out_cnt:^9} | {n_cnt:^9}")

    return [c_stats, out_stats, neg_stats]

def split_dataset(pretokenized_dir, tokenizer, max_len, dataset_ids, dataset_facts, dataset_arguments, dataset_claims, dataset_outcomes):
    train_ids, train_facts, train_arguments, train_claims, train_outcomes = [], [], [], [], []
    val_ids, val_facts, val_arguments, val_claims, val_outcomes = [], [], [], [], []
    test_ids, test_facts, test_arguments, test_claims, test_outcomes = [], [], [], [], []

    dataset_ids = [str(id) for id in dataset_ids]

    r_s = 42
    X_ids, X_test_ids, y_outcome, y_test_outcome = train_test_split(dataset_ids, dataset_outcomes, test_size=0.10,
                                                                    random_state=r_s)
    X_train_ids, X_valid_ids, y_train_outcome, y_valid_outcome = train_test_split(X_ids, y_outcome, test_size=0.10,
                                                                    random_state=r_s)

    case_dic = dict(zip(dataset_ids, zip(dataset_facts, dataset_claims, dataset_outcomes, dataset_arguments)))

    for id, value in tqdm(zip(case_dic.keys(), case_dic.values())):
        if id in X_train_ids:
            train_ids.append(id)
            train_facts.append(value[0])
            train_claims.append(value[1])
            train_outcomes.append(value[2])
            train_arguments.append(value[3])
        elif id in X_valid_ids:
            val_ids.append(id)
            val_facts.append(value[0])
            val_claims.append(value[1])
            val_outcomes.append(value[2])
            val_arguments.append(value[3])
        else:
            test_ids.append(id)
            test_facts.append(value[0])
            test_claims.append(value[1])
            test_outcomes.append(value[2])
            test_arguments.append(value[3])

    mlb = MultiLabelBinarizer()
    train_claims, train_outcomes = binarizer(train_claims, train_outcomes, mlb, True)
    test_claims, test_outcomes = binarizer(test_claims, test_outcomes, mlb)
    val_claims, val_outcomes = binarizer(val_claims, val_outcomes, mlb)

    print(
        f"{'split':^9} | {'claims':^9} | {'positives':^9} | {'negatives':^9}")
    training = data_stats(train_claims, train_outcomes, "train")
    validation = data_stats(val_claims, val_outcomes, "val")
    test = data_stats(test_claims, test_outcomes, "test")

    for i in [training, validation, test]:
        for j in i:
            print(j)

    print('Tokenizing data...')

    Path(pretokenized_dir).mkdir(parents=True, exist_ok=True)

    train_facts, train_masks = preprocessing_for_bert(train_facts, tokenizer, max=max_len)
    train_arguments, train_masks_arguments = preprocessing_for_bert(train_arguments, tokenizer, max=max_len)

    with open(pretokenized_dir + "/tokenized_train.pkl", "wb") as f:
        pickle.dump([train_facts, train_masks, train_arguments, train_masks_arguments, train_ids, train_claims, train_outcomes, mlb], f, protocol=4)

    val_facts, val_masks = preprocessing_for_bert(val_facts, tokenizer, max=max_len)
    val_arguments, val_masks_arguments = preprocessing_for_bert(val_arguments, tokenizer, max=max_len)

    with open(pretokenized_dir + "/tokenized_dev.pkl", "wb") as f:
        pickle.dump([val_facts, val_masks, val_arguments, val_masks_arguments, val_ids, val_claims, val_outcomes, mlb], f, protocol=4)

    test_facts, test_masks = preprocessing_for_bert(test_facts, tokenizer, max=max_len)
    test_arguments, test_masks_arguments = preprocessing_for_bert(test_arguments, tokenizer, max=max_len)

    with open(pretokenized_dir + "/tokenized_test.pkl", "wb") as f:
        pickle.dump([test_facts, test_masks, test_arguments, test_masks_arguments, test_ids, test_claims, test_outcomes, mlb], f, protocol=4)

    return train_ids, train_facts, train_claims, train_outcomes


def binarizer(claims, outcomes, mlb, fit=False):
    if fit:
        claims = mlb.fit_transform(claims)
        outcomes = mlb.transform(outcomes)
    else:
        claims = mlb.transform(claims)
        outcomes = mlb.transform(outcomes)

    return claims, outcomes

def preprocessing_for_bert(data, tokenizer, max=512):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    # For every sentence...
    input_ids = []
    attention_masks = []

    for sent in tqdm(data):
        sent = " ".join(sent)
        sent = sent[:500000] # Speeds the process up for documents with a lot of precedent we would truncate anyway.
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            truncation=True,
        )

        # Add the outputs to the lists
        input_ids.append([encoded_sent.get('input_ids')])
        attention_masks.append([encoded_sent.get('attention_mask')])

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def outcome_preprocess():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    get_data("ECHR/precedent/legal_bert", tokenizer, 512)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    get_data("ECHR/precedent/bert", tokenizer, 512)
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    get_data("ECHR/precedent/longformer", tokenizer, 4096)

def get_allowed(arts):
    allowed = ['10', '11', '13', '14', '18', '2', '3', '4', '5', '6', '7', '8', '9', 'P1-1', 'P4-2', 'P7-1', 'P7-4']
    new = []
    for i in arts:
        if i in allowed:
            new.append(i)
    return new


def simple_data(path="dataset/dev.jsonl"):

    all_facts, all_claims, all_outcomes, all_ids = [], [], [], []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        claims = get_allowed(result["allegedly_violated_articles"])
        outcomes = get_allowed(result["violated_articles"])
        if len(claims) >= len(outcomes) and len(claims) > 0:
            all_facts.append(result["facts"])
            all_claims.append(claims)
            all_outcomes.append(outcomes)
            all_ids.append(result["case_no"])

    return all_facts, all_claims, all_outcomes, all_ids

def simple_split(pretokenized_dir, tokenizer, max_len):
    val_facts, val_claims, val_outcomes, val_ids = simple_data("ECHR/alleged/dev.jsonl")
    test_facts, test_claims, test_outcomes, test_ids = simple_data("ECHR/alleged/test.jsonl")
    train_facts, train_claims, train_outcomes, train_ids = simple_data("ECHR/alleged/train.jsonl")

    mlb = MultiLabelBinarizer()
    train_claims, train_outcomes = binarizer(train_claims, train_outcomes, mlb, True)
    test_claims, test_outcomes = binarizer(test_claims, test_outcomes, mlb)
    val_claims, val_outcomes = binarizer(val_claims, val_outcomes, mlb)

    print(
        f"{'split':^9} | {'claims':^9} | {'positives':^9} | {'negatives':^9}")
    training = data_stats(train_claims, train_outcomes, "train")
    validation = data_stats(val_claims, val_outcomes, "val")
    test = data_stats(test_claims, test_outcomes, "test")

    for i in [training, validation, test]:
        for j in i:
            print(j)

    train_facts, train_masks = preprocessing_for_bert(train_facts, tokenizer, max=max_len)
    train_arguments, train_masks_arguments = train_facts, train_masks

    with open(pretokenized_dir + "/tokenized_train.pkl", "wb") as f:
        pickle.dump(
            [train_facts, train_masks, train_arguments, train_masks_arguments, train_ids, train_claims, train_outcomes,
             mlb], f, protocol=4)

    val_facts, val_masks = preprocessing_for_bert(val_facts, tokenizer, max=max_len)
    val_arguments, val_masks_arguments = val_facts, val_masks

    with open(pretokenized_dir + "/tokenized_dev.pkl", "wb") as f:
        pickle.dump([val_facts, val_masks, val_arguments, val_masks_arguments, val_ids, val_claims, val_outcomes, mlb],
                    f, protocol=4)

    test_facts, test_masks = preprocessing_for_bert(test_facts, tokenizer, max=max_len)
    test_arguments, test_masks_arguments = test_facts, test_masks

    with open(pretokenized_dir + "/tokenized_test.pkl", "wb") as f:
        pickle.dump(
            [test_facts, test_masks, test_arguments, test_masks_arguments, test_ids, test_claims, test_outcomes, mlb],
            f, protocol=4)

def chalkidis_preprocess():
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    simple_split("ECHR/alleged/legal_bert", tokenizer, 512)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    simple_split("ECHR/alleged/bert", tokenizer, 512)
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    simple_split("ECHR/alleged/longformer", tokenizer, 4096)


if __name__ == '__main__':
    chalkidis_preprocess()
    outcome_preprocess()
