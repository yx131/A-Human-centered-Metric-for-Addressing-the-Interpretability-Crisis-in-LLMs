import sys, pickle, re
sys.path.insert(0, '../Preprocess')
from preprocess_funcs import get_vectorizer_idx_to_word_mapping, prepare_text, prepare_text_view_friendly
import pandas as pd
import numpy as np
from sklearn import utils
import torch
from nltk.tokenize import word_tokenize
random_state = 13

########sampling function(s)
def my_completely_random_sample(df, num_sample=50):
    RS = np.random.RandomState(random_state)
    pos_idx = RS.randint(low=0, high=12499, size=num_sample) #assume the first 12500 rows of the df are pos
    neg_idx = RS.randint(low=12500, high=24999, size=num_sample) #assume the last 12500 rows of the df are pos
    pos_samples = df.loc[pos_idx][['review', 'sentiment']].values
    neg_samples = df.loc[neg_idx][['review', 'sentiment']].values
    all_samples = np.concatenate([pos_samples, neg_samples], axis=0)
    all_indices = np.concatenate([pos_idx, neg_idx], axis=0)
    all_samples_permuted, all_indices_permuted = utils.shuffle(all_samples, all_indices, random_state=random_state)
    reviews_permuted = list(all_samples_permuted[:, 0])
    targets_permuted = list(all_samples_permuted[:, 1])
    all_indices_permuted = list(all_indices_permuted)

    return reviews_permuted, targets_permuted, all_indices_permuted


def load_prev_samples(df, fname):
    with open(fname, 'rb') as f:
        prev_indices = pickle.load(f)['indices']
        print(f'previous indicies {prev_indices}')
        reviews_raw = df.loc[prev_indices, 'review'].values
        targets = df.loc[prev_indices, 'sentiment'].values
        return reviews_raw, targets, prev_indices


########sampling functions end

##################################################
#functions in this section work with attributions returned by captum
def see_att_vals(attribution, ID=0):
    ret = []
    for k in attribution.nonzero():
        ret.append(attribution[0, k[1]].item())
    print(f'{ID} {ret}')

#normalize positives/negatives separately
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

#normalize pos/neg by the abs max and neg
def attr_normalizing_func_2(attribution):
    # normlalizing negative values between -1 and 0 | normalizing positive values between 0 and 1
    # but make sure 0's stay 0
    attr_abs = torch.abs(attribution)
    pos_attr = torch.where(attribution > 0, attribution, torch.zeros(size=(1,)))
    neg_attr = torch.where(attribution < 0, attribution, torch.zeros(size=(1,)))
    neg_neg = -(neg_attr)

    pos_normal = (pos_attr - attr_abs.min()) / (attr_abs.max() - attr_abs.min())
    pos_normal -= attr_abs.min()  # so 0's stay 0
    neg_normal = ((neg_neg - attr_abs.min()) / (attr_abs.max() - attr_abs.min()))
    neg_normal -= attr_abs.min()
    neg_normal *= -1

    normalized_first_pos = torch.where(attribution > 0, pos_normal, attribution)
    normalized = torch.where(normalized_first_pos < 0, neg_normal, normalized_first_pos)

    return normalized

#nor normalizing
def attr_normalizing_func_3(attribution):
    return  attribution

def get_attributed_words(attribution):
    idx_dict = get_vectorizer_idx_to_word_mapping()
    words_attr = {}
    for l in attribution.nonzero():
        idx = l[1].item()  # l[1] here because the 0'th dimension is expected to be 0 since there's suppose to be 1 sample
        word, attr = idx_dict[idx], attribution[0, idx]
        words_attr[word] = attr
    return words_attr


#function to proc a word so that it matches the output of the preprocessing / the input to the model
def preprocess_word_proc_func(word):
    proc_w = prepare_text(word)
    return proc_w

#function to process a word so that it is pleasing to the eye. (To be used for display later)
HTML_brackets = re.compile(r'(\<)|(\/\>)')
HTML_br_enum = re.compile(r'^br$|\<br')
HEX_encoding = re.compile(r'[^\x00-\x7f]')
def prep_text_for_view(raw_review):
    preped_text = []
    for token in raw_review.split():
        token = HTML_br_enum.sub('', token)
        token = HTML_brackets.sub('', token)
        token = HEX_encoding.sub('', token)
        preped_text.append(token)
    return preped_text

#find indices of "words" that are not going to displayed. (bascially empty strings)
#if (on the odd chance) that they have attribution, their attribution wll be erased
def non_display_index(raw_review, word=''):
    words_for_view = prep_text_for_view(raw_review)
    no_dis_idx_list = [i for i in range(len(words_for_view)) if words_for_view[i] == word]
    return no_dis_idx_list

def assign_attr_to_word_idx(raw_review, words_attr):
    words_raw_input = raw_review.split() #word_tokenize(raw_review)
    # print(f'words_raw_input {words_raw_input}')
    attri_tensor = torch.zeros(size=(len(words_raw_input),))
    for word, attr in words_attr.items():
        index_pos_list = [i for i in range(len(words_raw_input)) if preprocess_word_proc_func(words_raw_input[i]) == word]
        attri_tensor[index_pos_list] = attr

    #zero out no displays
    no_dis_idx_list = non_display_index(raw_review)
    attri_tensor[no_dis_idx_list] = 0
    return attri_tensor


def conv_input_attri_to_word_attri(attribution, raw_review):
    # attribution = attr_normalizing_func(attribution)
    attribution = attr_normalizing_func_3(attribution)
    words_attr = get_attributed_words(attribution)
    # print(f'words attr: {words_attr}')
    attri_tensor = assign_attr_to_word_idx(raw_review, words_attr)
    return attri_tensor, words_attr



#attribution helper functions end
#######################################################################


#######################################################################
####functions for upkeeping
def collect_info_for_metric(model_out_list, model_out, attr_list, attr, wrd_attr_list, wrd_attr_dict):
    model_out_list.append(model_out)
    attr_list.append(attr)
    wrd_attr_list.append(wrd_attr_dict)
    return model_out_list, attr_list, wrd_attr_list

def save_info(indices, raw_reviews, targets, model_out_list, attr_list, wrd_attr_list, fname='framework_info_saved.pkl'):
    to_save = {}
    to_save['indices'] = indices
    to_save['raw_reviews'] = raw_reviews
    to_save['targets'] = targets
    to_save['model_out_list'] = model_out_list
    to_save['attr_list'] = attr_list
    to_save['wrd_attr_list'] = wrd_attr_list

    with open(fname, 'wb') as f:
        pickle.dump(to_save, f)

    return to_save

