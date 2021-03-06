# coding=utf8
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import random
import re
from zhon.hanzi import punctuation
import jiagu
# import logging
# import synonyms
from tqdm import trange
from tqdm import tqdm
import time

import os

import pandas as pd
from pypinyin import pinyin, lazy_pinyin, Style
import naming
# import tensorflow as tf

# from sklearn import feature_extraction
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer


def _showPlt(v):
    pass
    # Plot the values as a histogram to show their distribution.
    # plt.figure(figsize=(10,10))
    # plt.hist(v, bins=200)
    # plt.show()


def _nlp():
    pass
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # text = "Here is the sentence I want embeddings for."
    # text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    # marked_text = "[CLS] " + text + " [SEP]"
    # print (marked_text)
    # tokenized_text = tokenizer.tokenize(marked_text)
    # print (tokenized_text)
    # list(tokenizer.vocab.keys())[5000:5020]
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # for tup in zip(tokenized_text, indexed_tokens):
    #   print (tup)
    #
    # segments_ids = [1] * len(tokenized_text)
    # print (segments_ids)
    #
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    #
    # # Load pre-trained model (weights)
    # model = BertModel.from_pretrained('bert-base-uncased')
    #
    # # Put the model in "evaluation" mode, meaning feed-forward operation.
    # model.eval()
    #
    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # print ("Number of layers:", len(encoded_layers))
    # layer_i = 0
    #
    # print ("Number of batches:", len(encoded_layers[layer_i]))
    # batch_i = 0
    #
    # print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    # token_i = 0
    #
    # print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))
    #
    # token_i = 5
    # layer_i = 5
    # vec = encoded_layers[layer_i][batch_i][token_i]
    #
    #
    # # Convert the hidden state embeddings into single token vectors
    #
    # # Holds the list of 12 layer embeddings for each token
    # # Will have the shape: [# tokens, # layers, # features]
    # token_embeddings = []
    #
    # # For each token in the sentence...
    # for token_i in range(len(tokenized_text)):
    #
    #     # Holds 12 layers of hidden states for each token
    #     hidden_layers = []
    #
    #     # For each of the 12 layers...
    #     for layer_i in range(len(encoded_layers)):
    #         # Lookup the vector for `token_i` in `layer_i`
    #         vec = encoded_layers[layer_i][batch_i][token_i]
    #
    #         hidden_layers.append(vec)
    #
    #     token_embeddings.append(hidden_layers)
    #
    # # Sanity check the dimensions:
    # print("Number of tokens in sequence:", len(token_embeddings))
    # print("Number of layers per token:", len(token_embeddings[0]))
    # concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]
    #
    # summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]
    #
    # sentence_embedding = torch.mean(encoded_layers[11], 1)
    # print ("Our final sentence embedding vector of shape:"), sentence_embedding[0].shape[0]
    #
    # for i,x in enumerate(tokenized_text):
    #   print (i,x)
    #
    # print ("First fifteen values of 'bank' as in 'bank robber':")
    # summed_last_4_layers[10][:15]
    #
    #
    # print ("First fifteen values of 'bank' as in 'bank vault':")
    # summed_last_4_layers[6][:15]
    #
    # print ("First fifteen values of 'bank' as in 'river bank':")
    # summed_last_4_layers[19][:15]
    #
    # from sklearn.metrics.pairwise import cosine_similarity
    #
    # # Compare "bank" as in "bank robber" to "bank" as in "river bank"
    # different_bank = cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers[19].reshape(1,-1))[0][0]
    #
    # # Compare "bank" as in "bank robber" to "bank" as in "bank vault"
    # same_bank = cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers[6].reshape(1,-1))[0][0]
    #
    # print ("Similarity of 'bank' as in 'bank robber' to 'bank' as in 'bank vault':",  same_bank)
    #
    # print ("Similarity of 'bank' as in 'bank robber' to 'bank' as in 'river bank':",  different_bank)


def _distance(family_name, names):
    """
    ???????????????????????????
    :param family_name:
    :param names:
    :return:
    """
    pass


def _sentiment(family_name, names):
    """
    ??????????????????
    :param family_name:
    :param names:
    :return:
    """
    pass


def _stoke(family_name, names):
    """
    ????????????
    :param family_name:
    :param names:
    :return:
    """
    pass


def _similar(name_dim, family_name):
    """
    ??????????????????????????????
    :param name_dim:
    :param family_name:
    :return:
    """
    pass


def _pronounce(name_dim, family_name):
    """
    ??????????????????
    :param name_dim:
    :param family_name:
    :return:
    """
    pass


def _tone(name_dim, family_name):
    """
    ????????????
    :param name_dim:
    :param family_name:
    :return:
    """
    pass


def _random_select(name_dim):
    """
    ?????????????????????????????????????????????
    :return:
    """
    # name = torch.tensor(name_dim[:][0])
    # name_set = set(name)
    # random.randrange(1,len(name_set))
    pass


def generate_name(name_dim, family_name, topK=10, sample_count=1000):
    """
    :param name_dim: ????????????
    :param family_name: ??????
    :param topK: ???TOP
    :param sample_count: ???????????? 0??????????????????

    ??????????????????????????????????????????????????????????????????
    ?????????????????????
    ?????????????????????????????????????????????????????????
    1. ??????????????????????????????????????????????????????????????????????????????[D] ???????????????????????? d??????????????????????????????????????????????????????
    "?????????????????????????????????????????????????????????????????????????????????????????????"??????????????????????????????????????????????????????????????????
    ?????????"????????????????????????????????????????????????????????????"????????????????????????????????????????????????

    2. ????????????????????????????????????????????? ???????????? ??????[E] ????????????????????????????????? e
    ??????????????? ?????? ????????????????????? "????????????"?????????????????????0.93???????????????????????????????????????"????????????"??????????????????0.5?????????????????????
    ??????????????????????????????
    3. ??????????????????????????? ??????????????????[S] ???????????????????????? s???????????????????????????????????????????????????????????????????????????????????????
    4. ??????????????????????????????????????????????????????[Y] ???????????????????????? y???????????????????????????????????????????????????????????????
    ????????????????????????????????????"???????????????"??????????????????????????????"???"?????????"???"?????????????????????????????????????????????????????????????????????
    5. ???????????????????????????????????????????????????[P] ???????????????????????? p????????????
    5.1 ????????????P????????????????????????????????????????????????????????????????????????????????????????????????????????????3????????????1???3???????????????3?????????????????????
    6 ?????????????????????????????????????????????[T] ????????????????????????t,?????????
    6.1 ???????????????????????????????????????????????????????????????
    6.2 ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????3*3 = 9???????????????????????????????????????????????????
        ???????????????????????????????????????A42A21???????????????????????????10???????????????????????????4????????????????????????40???????????????????????????????????????
        ??????????????????????????????????????????????????????????????????
    6.3 ???????????????????????????????????????
    12???21???13???31???14???41???23???32???34???43
1   1   2   3  4   5   6  7   8   9  10
2   1   2   3  4   5   6  7   8   9  10
3   1   2   3  4   5   6  7   8   9  10
4   1   2   3  4   5   6  7   8   9  10
1????????????2????????????3?????????4??????
??????????????????????????????????????????
11???12???13
21???23???24
31???32???34
41???42???43
?????????????????????????????????????????????
??????2????????????1?????????????????????
4????????????????????????????????????????????????4???
121,133,142??????|124,132,141??????
212,213,214,231,232,241,242,243??????|221,224,223,211,234
311,312,313,314,321,324,341,342,343??????|322,323??????
411,412,413,414,421,423,424,431,432,434??????
    7. ??? d,e,s,y,p,t ?????????????????? [N] ????????????????????????n
    ??????N?????????????????????????????????????????????
    8. n???????????????????????????????????????????????????????????????
    ?????????D,E,S,Y,P,N??????????????????
    9. ???????????????????????????????????????????????????????????????????????? ???????????????????????????????????????
    :param name_dim: ????????????
    :param family_name: ??????
    :param topK ??????????????????
    :return:
    """
    d = 0
    e = 0
    s = 0
    y = 0
    p = 0
    t = 0
    # x1 = random.randint()
    # x2 = random.randint()
    # x3 = random.randint()
    # x4 = random.randint()
    # x5 = random.randint()
    # x6 = random.randint()
    bestNounce = [11, 12, 13, 14, 21, 23, 24, 31, 32, 34]
    # ????????????
    columns = ['character', 'sentence', 'line', 'document', 'shengmu',
               'yunmu', 'shengdiao', 'bihua', 'cixing','char_sentiment',
               'sentiment_score', 'sentence_length', 'td_idf', 'degree', 'char_pos',
               'pos','is_begin_of_sent','is_end_of_sent','is_begin_of_line','is_end_of_line']
    #?????????
    counts = 0
    #????????????
    excel = pd.DataFrame(pd.read_csv(name_dim))
    # ???????????????
    family_name_shengmu = pinyin(family_name, style=Style.INITIALS)[0][0]
    # ???????????????
    family_name_yunmu = pinyin(family_name, style=Style.FINALS)[0][0]
    # ?????????????????????????????????
    firstName = excel[(((excel["shengdiao"] == 1) | (excel["shengdiao"] == 2) | (excel["shengdiao"] == 3)) & (
                excel["shengmu"] != family_name_shengmu) & (excel["yunmu"] != family_name_yunmu))]
    # ?????????????????????????????????
    secondName = excel[(excel["shengdiao"] != 4)& ((excel["shengmu"] != family_name_shengmu) & (excel["yunmu"] != family_name_yunmu))]
    result = {}

    firstNameCache = []
    secondNameCache = []
    if sample_count > 0:
        firstName = firstName.sample(n=sample_count)
        secondName = secondName.sample(n=sample_count)
    with tqdm(total=len(firstName)) as pbar:
        pbar.set_description('Processing:')
        for i, f in firstName.iterrows():
            pbar.update(1)
            # ??????????????????????????????
            if f["character"] in firstNameCache:
                continue
            # ??????????????????
            if f["character"] in naming.bad_name:
                continue
            # ?????????????????????????????????
            firstNameCache.append(f["character"])
            for j, s in secondName.iterrows():
                # ??????????????????
                if s["character"] in naming.bad_name:
                    continue
                mean_score = 0
                bihua_score = 0
                nounce_score = 0
                sum_score = 0
                if f["character"] == s["character"]:
                    continue
                preTest = family_name + f["character"] + s["character"]
                if preTest in result.keys():
                    continue
                """
                ????????????????????????
                """
                if f["line"] == s["line"]:
                    mean_score += 10
                    """
                        ??????????????????????????????
                    """
                    if f["is_begin_of_line"] and s["is_end_of_line"]:
                        mean_score += 10
                    elif f["is_end_of_line"] and s["is_begin_of_line"]:
                        mean_score += 5

                """
                ?????????????????????
                """
                if f["sentence"] == s["sentence"]:
                    mean_score += 10
                    """
                    ?????????????????????
                    """
                    if f["is_begin_of_sent"] and s["is_end_of_sent"]:
                        mean_score += 20
                    elif f["is_end_of_sent"] and s["is_begin_of_sent"]:
                        mean_score +=15
                bihua_score = 100 / (f["bihua"] + s["bihua"])
                if (f["shengdiao"] * 10 + s["shengdiao"]) in bestNounce:
                    nounce_score = 10
                sum_score = mean_score * 100 + nounce_score * 10 + bihua_score * 5
                # if sum_score < 1100:
                #     continue
                score=[]
                for c in columns:
                    score.append(f[c])
                for c in columns:
                    score.append(s[c])
                score.append(sum_score)

                # result[preTest] = [f["character"], s["character"], sum_score,f['sentence'], f['line'], f['document'],s['sentence'], s['line'], s['document']]
                result[preTest] = score
                counts += 1
                if 0 < topK < counts:
                    return result
    return result


"""
??????
"""
def save_name_to_csv(line, file_name):
    """
        ?????????CSV
        :param line: ??????????????????
        :param file_name: ?????????
        :return:
        """
    columns = ['name', '???1', '???2', 'score','???1?????????','???1?????????','???1?????????','???2?????????','???2?????????','???2?????????']
    tmp = []
    for k in line.keys():
        names = []
        names.append(k)
        for c in line.get(k):
            names.append(c)
        tmp.append(names)
        # tmp.append([k, line.get(k)[0],
        #             line.get(k)[1], line.get(k)[2],
        #             line.get(k)[3],line.get(k)[4],
        #             line.get(k)[5],line.get(k)[6],
        #             line.get(k)[7],line.get(k)[8]])
    # exefile = pd.DataFrame(tmp, columns=columns)
    exefile = pd.DataFrame(tmp)
    exefile.to_csv(file_name, index=0, encoding='utf_8_sig')


if __name__ == "__main__":
    namemix = generate_name('./??????.csv', "???", -1,-1)
    save_name_to_csv(namemix, "./??????_??????.csv")
