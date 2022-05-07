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
    计算姓和名字的距离
    :param family_name:
    :param names:
    :return:
    """
    pass


def _sentiment(family_name, names):
    """
    计算情感分数
    :param family_name:
    :param names:
    :return:
    """
    pass


def _stoke(family_name, names):
    """
    计算笔画
    :param family_name:
    :param names:
    :return:
    """
    pass


def _similar(name_dim, family_name):
    """
    计算姓和名字的相似度
    :param name_dim:
    :param family_name:
    :return:
    """
    pass


def _pronounce(name_dim, family_name):
    """
    计算声母韵母
    :param name_dim:
    :param family_name:
    :return:
    """
    pass


def _tone(name_dim, family_name):
    """
    计算声调
    :param name_dim:
    :param family_name:
    :return:
    """
    pass


def _random_select(name_dim):
    """
    随机选择一些字，作为名字的种子
    :return:
    """
    # name = torch.tensor(name_dim[:][0])
    # name_set = set(name)
    # random.randrange(1,len(name_set))
    pass

"""
完成
"""
def generate_name(name_dim, family_name, topK=10):
    """

    生成一些名字，进行组合。这里是最核心的算法。
    模型算法规则：
    随机选两个字，按照如下规则进行计算打分
    1. 若两个名字在一句话中，或者是一行中，通过一个距离算法[D] 得出一个距离系数 d，距离越近，表明两个字约有含义。例如
    "绸缪束刍，三星在隅。今夕何夕，见此邂逅？子兮子兮，如此邂逅何？"隅和夕在同一段文字中，这表明两者关系比较近。
    再比如"菁菁者莪，在彼中陵。既见君子，锡我百朋。"中的菁菁距离更近，表明更有相关性

    2. 若名字所在的那两句话的情感系数 通过一个 算法[E] 得出一个新的情感系数分 e
    例如两个字 隅夕 ，其中隅字出自 "三星在隅"，其情感系数为0.93，是个非常正向额词。夕出自"今夕何夕"，情感系数为0.5，那么加载一起
    是个不错的正向分数。
    3. 将两个名字的笔画数 通过一个算法[S] 得出一个笔画系数 s。目前的规则是，两个字加起来的笔画越少越好。笔画越少分越高
    4. 将两个名字与姓氏，通过一个相似度算法[Y] 得出一个含义系数 y。规则含义：名字与姓氏之间相似度越低越好。
    越低表明两者越不容易出现"撞车的情况"，例如：如果有个人姓"王"，那么"侯"字与王字是近义词。两个这两个字组成在一起不好听
    5. 将两个名字与姓氏，通过一个发音算法[P] 得出一个发音系数 p。含义：
    5.1 发音算法P的规则是：姓氏和名字的声母、韵母尽量避免相同。姓氏和名字避免声调一致，若3字的可以1，3同调，或者3个字都不同调。
    6 将名字与姓氏，通过一个声调算法[T] 得到一个声调系数t,含义：
    6.1 若两个字是叠音，只要跟姓氏不同音，就可以。
    6.2 在三个字都不同调的情况下，姓氏位置不能变，两个字的名字全排列组合再去掉声调一致的组合大致有3*3 = 9种组合。其中声调相同只适用于叠字。
        因此若不考虑叠字，则应该有A42A21种组合，也就是说有10种组合。配合姓氏的4中声调，应该是有40中排列组合。应该针对每一个
        姓氏声调进行单独排列。这种声调矩阵，需要计算
    6.3 名字与姓氏之间的声调关系：
    12，21，13，31，14，41，23，32，34，43
1   1   2   3  4   5   6  7   8   9  10
2   1   2   3  4   5   6  7   8   9  10
3   1   2   3  4   5   6  7   8   9  10
4   1   2   3  4   5   6  7   8   9  10
1：阴平；2：阳平；3：上；4：去
单字名字规律，第一个是姓氏：
11，12，13
21，23，24
31，32，34
41，42，43
两个字名字规律：第一个是姓氏：
最佳2分，次之1分，其余不得分
121,133,142最佳|124,132,141次之
212,213,214,231,232,241,242,243最佳|221,224,223,211,234
311,312,313,314,321,324,341,342,343最佳|322,323次之
411,412,413,414,421,423,424,431,432,434最佳
    7. 将 d,e,s,y,p,t 通过一个算法 [N] 得出一个综合系数n
    算法N中各个系数需要通过深度学习得到
    8. n越大表明越好：优化参数和损失函数要好好设计
    超参：D,E,S,Y,P,N中的算法参数
    9. 姓氏与名字，应该符合开口音与闭口音规则，即姓名名 应该符合开闭开，或者闭开闭
    :param name_dim: 名字矩阵
    :param family_name: 姓氏
    :param topK 返回几个名字
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
    # 打印的列
    columns = ['character', 'sentence', 'line', 'document', 'shengmu',
               'yunmu', 'shengdiao', 'bihua', 'cixing','char_sentiment',
               'sentiment_score', 'sentence_length', 'td_idf', 'degree', 'char_pos',
               'pos','is_begin_of_sent','is_end_of_sent','is_begin_of_line','is_end_of_line']
    #计数器
    counts = 0
    #读取信息
    excel = pd.DataFrame(pd.read_csv(name_dim))
    # 姓氏的声母
    family_name_shengmu = pinyin(family_name, style=Style.INITIALS)[0][0]
    # 姓氏的韵母
    family_name_yunmu = pinyin(family_name, style=Style.FINALS)[0][0]
    # 找出姓名中的第一个名字
    firstName = excel[(((excel["shengdiao"] == 1) | (excel["shengdiao"] == 2) | (excel["shengdiao"] == 3)) & (
                excel["shengmu"] != family_name_shengmu) & (excel["yunmu"] != family_name_yunmu))]
    # 找出姓名中的第二个名字
    secondName = excel[((excel["shengmu"] != family_name_shengmu) & (excel["yunmu"] != family_name_yunmu))]
    result = {}

    firstNameCache = []
    secondNameCache = []
    firstName = firstName.sample(n=1000)
    secondName = secondName.sample(n=1000)
    with tqdm(total=len(firstName)) as pbar:
        pbar.set_description('Processing:')
        for i, f in firstName.iterrows():
            pbar.update(1)
            if f["character"] in firstNameCache:
                continue
            if f["character"] in naming.bad_name:
                continue
            firstNameCache.append(f["character"])
            for j, s in secondName.iterrows():
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
                如果是同一篇文章
                """
                if f["line"] == s["line"]:
                    mean_score += 10
                    """
                        如果是文章的一头一尾
                    """
                    if f["is_begin_of_line"] and s["is_end_of_line"]:
                        mean_score += 10
                    elif f["is_end_of_line"] and s["is_begin_of_line"]:
                        mean_score += 5

                """
                如果是同一段话
                """
                if f["sentence"] == s["sentence"]:
                    mean_score += 10
                    """
                    如果是一头一尾
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
完成
"""
def save_name_to_csv(line, file_name):
    """
        保存到CSV
        :param line: 要保存的内容
        :param file_name: 文件名
        :return:
        """
    columns = ['name', '名1', '名2', 'score','名1出自句','名1出自章','名1出自篇','名2出自句','名2出自章','名2出自篇']
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


def _loss(name_dim):
    """
    损失函数
    :param name_dim:
    :return:
    """
    pass


def _sgd(name_dim):
    """
    定义优化函数
    :param name_dim:
    :return:
    """
    pass


def _train(name_dim):
    """
    训练
    :param name_dim:
    :return:
    """
    pass


if __name__ == "__main__":
    namemix = generate_name('./庄子.csv', "鲍", -1)
    save_name_to_csv(namemix, "./庄子_姓名.csv")
