import sys, pickle, re, time
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
random_state = 666

#Section of random sampling function starts here
##############################################################################
def sample_random_glue_sst2(num_sample=50):
    sst2 = load_dataset("glue", "sst2")
    sst2_train = sst2['train']
    max_sample = sst2_train.num_rows

    RS = np.random.RandomState(random_state)
    pos_rows = sst2['train'].filter(lambda x: x['label']==1)
    neg_rows = sst2['train'].filter(lambda x: x['label']==0)
    pos_idx, neg_idx = pos_rows['idx'], neg_rows['idx']

    pos_sample_idx = RS.choice(pos_idx, size=int(num_sample/2), replace=False)
    neg_sample_idx = RS.choice(neg_idx, size=int(num_sample/2), replace=False)
    all_sample_idx = np.concatenate((pos_sample_idx, neg_sample_idx))
    RS.shuffle(all_sample_idx) #this method shuffles inplace
    all_samples_selected = sst2_train.select(all_sample_idx)
    sentences, labels, idx = all_samples_selected['sentence'], all_samples_selected['label'], all_samples_selected['idx']
    
    return sentences, labels, idx

##################################################################################
def sample_random_glue_stsb(num_sample=50):
    stsb = load_dataset("glue", "stsb")
    stsb_val = stsb['validation']
    max_sample = stsb_val.num_rows
    RS = np.random.RandomState(random_state)
    all_sample_idx = RS.choice(stsb_val['idx'], size=num_sample, replace=False)
    print(all_sample_idx)
    all_samples_selected = stsb_val.select(all_sample_idx)
    sentence_pairs, labels, idx = [[s['sentence1'], s['sentence2']] for s in all_samples_selected],\
                                    all_samples_selected['label'], \
                                    all_samples_s

##################################################################################
def sample_random_glue_qnli(num_sample=50):
    qnli = load_dataset("glue", "qnli")
    qnli_val = qnli['validation']
    max_sample = qnli_val.num_rows

    RS = np.random.RandomState(random_state)
    ent_rows = qnli_val.filter(lambda x: x['label'] == 0)
    nonent_rows = qnli_val.filter(lambda x: x['label'] == 1)
    ent_idx, nonent_idx = ent_rows['idx'], nonent_rows['idx']

    ent_sample_idx = RS.choice(ent_idx, size=int(num_sample / 2), replace=False)
    nonent_sample_idx = RS.choice(nonent_idx, size=int(num_sample / 2), replace=False)
    all_sample_idx = np.concatenate((ent_sample_idx, nonent_sample_idx))
    RS.shuffle(all_sample_idx)  # this method shuffles inplace
    all_samples_selected = qnli_val.select(all_sample_idx)
    question_paragraph_pairs, labels, idx = [[s['question'], s['sentence']] for s in all_samples_selected],\
                                            [0 if l == 1 else 1 for l in all_samples_selected['label']],\
                                            all_samples_selected['idx']
    return question_paragraph_pairs, labels, idx
##################################################################################

#Section of random sampling functions end here





##################################################################################
def get_continuation_mapping(offset_mapping):
    conti_bools = []
    for i, m in enumerate(offset_mapping):
        if i == 0 or i == 1:
            conti_bools.append(False)
            continue
        else:
            if m[0] == offset_mapping[i-1][1]:
                conti_bools.append(True)
            else:
                conti_bools.append(False)
    return conti_bools

def get_continuous_attributions(conti_bools, word_attributions):
    conti_attri = []
    for i, (conti, attr) in enumerate(zip(conti_bools, word_attributions)):
        if conti == True:
            conti_attri[-1] = (conti_attri[-1] + attr) / 2 #take the average of the two attributions
        else:
            conti_attri.append(attr)
    print(f'word attr {word_attributions}')
    print(f'conti attr {conti_attri}')
    return torch.tensor(conti_attri)

def get_continuous_raw_inputs(conti_bools, detokenized):
    conti_raw = []
    for i, (conti, detok) in enumerate(zip(conti_bools, detokenized)):
        if conti == True:
            conti_raw[-1] = conti_raw[-1] + detok
        else:
            conti_raw.append(detok)
    print(f'detokenized {detokenized}')
    print(f'len conti_raw {len(conti_raw)}')
    print(f'conti_raw {conti_raw}')
    return conti_raw
##################################################################################

###############################################################################
def attr_normalizing_func(attribution):
    # normlalizing negative values between -1 and 0 | normalizing positive values between 0 and 1
    # but make sure 0's stay 0
    print(f'attr dtype {attribution.dtype}')
    pos_attr = torch.where(attribution > 0, attribution, torch.zeros(size=(1,)))
    neg_attr = torch.where(attribution < 0, attribution, torch.zeros(size=(1,)))
    neg_neg = -(neg_attr)

    pos_normal = (pos_attr - pos_attr.min()) / (pos_attr.max() - pos_attr.min())
    pos_normal -= pos_normal.min()  # so 0's stay 0
    neg_normal = ((neg_neg - neg_neg.min()) / (neg_neg.max() - neg_neg.min()))
    neg_normal -= neg_normal.min()
    neg_normal *= -1

    normalized_first_pos = torch.where(attribution > 0, pos_normal, attribution)
    normalized = torch.where(normalized_first_pos < 0, neg_normal, normalized_first_pos)

    return normalized
###############################################################################



#######################################################################
####functions for upkeeping
def collect_info_for_metric(model_out_list, model_out, raw_attr_list, raw_attr, conti_attr_list, conti_attr, raw_input_list, raw_input):
    model_out_list.append(model_out)
    raw_attr_list.append(raw_attr)
    conti_attr_list.append(conti_attr)
    raw_input_list.append(raw_input)
    return model_out_list, raw_attr_list, conti_attr_list, raw_input_list

def save_info(indices, raw_data, targets, model_out_list, raw_attr_list, conti_attr_list, raw_input_list, fname='framework_info_saved.pkl'):
    to_save = {}
    to_save['indices'] = indices
    to_save['raw_data'] = raw_data
    to_save['targets'] = targets
    to_save['model_out_list'] = model_out_list
    to_save['raw_attr_list'] = raw_attr_list
    to_save['conti_attr_list'] = conti_attr_list
    to_save['raw_input_list'] = raw_input_list

    with open(fname, 'wb') as f:
        pickle.dump(to_save, f)

    return to_save


from IPython.display import Javascript
from nbconvert import HTMLExporter
def save_notebook():
    display(
        Javascript("IPython.notebook.save_notebook()"),
        include=['application/javascript']
    )

def output_HTML(read_file, output_file):
    import codecs
    import nbformat
    exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    output_notebook = nbformat.read(read_file, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file, 'w', encoding='utf-8').write(output)


def download_HTML(cur_file_name, out_file_name):
    save_notebook()
    time.sleep(3)
    output_HTML(cur_file_name, out_file_name)