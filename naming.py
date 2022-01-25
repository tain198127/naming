#coding=utf-8
import re
import torch
import torch.nn as nn
from zhon.hanzi import punctuation
import jieba
import jieba.analyse
import jieba.posseg
import jiagu
import logging
import os
from torch.autograd import Variable
import pandas as pd
from pypinyin import pinyin, lazy_pinyin,Style
import tensorflow as tf

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

'''
1. 两个字的名字，第一个字要1声，第二个字，如果是a,o,e,eu,wu,yu的，二声好听。否则是一声好听。
2. 两个字的名义，必须是句子的开头，或者结尾，或者一头一尾。
3. 笔画要简单。
4. 要在多个文章中，同时出现在头或者尾，或者一头一尾。
5. 组合的名义，用jieba把词性弄出来，要偏正常的。也就是所谓的好词。
6. 名字首选形容词，定状补语。连词也不错。但是不要名词动词。
7. 这些字不能已经是现在常用的名义，也不能是跟他们的音相似的。
8. 姓名之间的特点是：姓和名之间必须是有一个起伏的，比如如果姓是4声，后面的名字就要1声再配一个1声或2声。如果姓是1声，那名字中第二个字就要变成2.3.4声，最后一个字又变回1声。
9. 鲍本身就是四声，而且是爆破音，因此后面的名字不能再出现爆破音。而且第一个字要是1声，第二个字可以是1声也可以是2声。

'''
word_to_ix = {"hello": 0, "world": 1}  # 用0代表hello，用1代表world
print(torch.LongTensor([0]))
# jieba.enable_paddle()
embeds = nn.Embedding(2, 5)  # 第一步定义词向量大小，参数一为单词个数，二为单词长度
"""
单字<-->分词<-->文件位置的map
"""

# character_idx = {}
"""
要跳过的词
"""
skip_word = punctuation+'□'
pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| ' \
          r'|…|（|）|，|。|；|：|？|（|）|【|】|「|」|！|、|\||《|》|\" |\" |在|兮|否'
strokes = []
#可以保留的韵母
yunmuStay=['a','o','e','eu','wu','yu']
#要跳过的声母，因为名字有爆破音，因此不需要再次出现爆破音
skip_shengmu=['b','p','t','d','k']
skip_yunmu=['ao']
#要跳过的词性，名词、代词、动词、量词都要跳过。因为这些词在名字中不好听
skip_cixing=['n','r','v','m','p','d','z']

def cut(line):
    split = re.split(pattern, line)
    return split


# 读文件
def readFile(fileName):
    word_idx = {}
    """
    :param fileName:
    seg_list:一句话
    line 行
    file 文章
    :return: 语句,行,文章名称的map
    """

    f = open(fileName)
    inserted_hash_tag=[]
    for line in f.readlines():
        if line in skip_word or line.isdecimal() or line.isdigit() or line.isspace():
            continue

        # seg_list = jieba.cut(line, use_paddle=True)
        seg_list = cut(line) #切割成一句话
        # 指的是哪篇文章的哪句话
        # word表示的是切割出来的词
        # line表示一行 fileName表示文章名称
        hash = str(line.__hash__())+'_'+str(fileName.__hash__())
        doc = {'line': line, 'file': fileName,'hash':hash}
        for word in seg_list:
            if word in skip_word or word.isdecimal() or word.isdigit() or word.isspace():
                continue
            if word not in word_idx.keys():
                word_idx[word] = []
            """倒排索引"""
            logging.debug("解析出来的seg_list的每一条信息是:%s",word)
            #去重
            docs = [d for d in word_idx.get(word) if d['hash'] == hash]
            if docs is None or len(docs) == 0:
                word_idx.get(word).append(doc)
    return word_idx

# 生成索引
def read_character(W_IDX,parentFirstName, isSkipBadWord=False):
    if parentFirstName is None:
        raise BaseException('parentFirstName 不能为空')
    first_name_shengmu = pinyin(parentFirstName, style=Style.INITIALS)[0][0]
    first_name_yunmu = pinyin(parentFirstName, style=Style.FINALS)[0][0]
    character_idx = {}
    """
    字符串-->句子-->行-->文章 的关联关系
    :param W_IDX:
    :param parentFirstName 父亲的姓氏
    :param isSkipBadWord 是否跳过不好的词，默认不跳过
    :return:
    """
    # word表示的是词
    for word in W_IDX:
        # char表示词中的每一个字
        # 情感

        for char in word:
            isSkip = False
            wordCut = jieba.posseg.cut(char)
            # 词性
            cixing = 'n'
            for w, f in wordCut:
                cixing = f
            # 如果词性是名词、动词等等，都跳过
            for c in skip_cixing:
                if c in cixing:
                    isSkip=True
                    continue
            if isSkip and isSkipBadWord:
                continue
            #如果是需要跳过的声母，则跳过

            #如果是空格、数字等等，重来

            if char in skip_word or char.isdecimal() or char.isdigit() or char.isspace():
                continue
            if char not in character_idx.keys():
                character_idx[char] = []
            doc_idx = W_IDX.get(word)
            # 拼音，声调，笔画数

            shengmu = pinyin(char, style=Style.INITIALS)[0][0]
            yunmu = pinyin(char, style=Style.FINALS)[0][0]
            if (shengmu ==first_name_shengmu or yunmu == first_name_yunmu) :
                continue
            if shengmu in skip_shengmu and isSkipBadWord:
                continue
            if yunmu in skip_yunmu and isSkipBadWord:
                continue
            #声调
            shengdiao=0
            if pinyin(char, style=Style.TONE2) and pinyin(char, style=Style.TONE2)[0][0] and re.findall(r'\d+', pinyin(char, style=Style.TONE2)[0][0]):
                shengdiao = re.findall(r'\d+', pinyin(char, style=Style.TONE2)[0][0])[0]
            #如果声调不是1，2声，则抛弃
            if shengdiao not in ['1','2'] and isSkipBadWord:
                continue
            #如果声调是2声，而且韵母不是a,o,e,i,wu,yu，则跳过
            if (shengdiao == 2 and yunmu not in yunmuStay) and isSkipBadWord:
                continue
            #笔画
            bihua = get_stroke(char)
            if bihua > 12 and isSkipBadWord:
                continue

            for w in doc_idx:
                # sentiment positive 还是negivate。以及对应的分数
                # 要看整个句子是不是好的句子，好句子才留下
                sentiment = jiagu.sentiment(w['line'])
                # 排除负能量词
                if (sentiment[0] == 'negative' or sentiment[1] < 0.75) and isSkipBadWord:
                    continue
                # 这里要计算字符的信息
                # 建议规则 笔画要少，
                # 词性要求必须是好的
                # 不要动名量词
                # 计算每个字在每句话中的position
                # 计算每个字在每句话中的TF-IDF
                node = {'sent': word,
                        'line': w['line'],
                        'doc': w['file'],
                        'shengmu':shengmu,
                        'yunmu':yunmu,
                        'shengdiao':shengdiao,
                        'bihua':bihua,
                        'cixing':cixing,
                        'sentiment':sentiment[0],
                        'sentiment_score':sentiment[1]
                        }
                character_idx.get(char).append(node)
    return character_idx

def _init_stroke(strokes_path):
    with open(strokes_path, 'r') as fr:
        for line in fr:
            strokes.append(int(line.strip()))

def get_stroke(c):
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段
    unicode_ = ord(c)
    if 13312 <= unicode_ <= 64045:
        return strokes[unicode_-13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes[unicode_-80338]
    else:
        -1
        # can also return 0

# 保存到csv
def save_character_to_csv(line, file_name):
    """
    保存character_idx到csv
    :param c_idx:
    :file_name 保存的文件位置
    :return:
    """
    columns = ['character', 'sentence', 'line', 'document','shengmu','yunmu','shengdiao','bihua','cixing','sentiment',
               'sentiment_score','td_idf','degree','char_pos','pos']
    # line = []
    # for c in c_idx:
    #     """
    #     拿到每一个字，和字对应的句子+文章
    #     """
    #     char_idxer = c_idx.get(c)
    #     for ch in char_idxer:
    #         """
    #         枚举每一个句子元数据，sent表示句子 file表示文章名
    #         """
    #         line.append([c, ch['sent'], ch['line'],ch['doc'],ch['shengmu'],ch['yunmu'],ch['shengdiao'],ch['bihua'],ch['cixing'],ch['sentiment'],ch['sentiment_score']])

    exefile = pd.DataFrame(line, columns=columns)
    exefile.to_csv(file_name, index=0, encoding='utf_8_sig')

# 对字进行过滤
def name_filter(name_dim):



    pass

# 转成N维数组
def name_mapping(c_idx):

    line = []
    for c in c_idx:
        """
        拿到每一个字，和字对应的句子+文章
        """
        char_idxer = c_idx.get(c)
        for ch in char_idxer:
            """
            枚举每一个句子元数据，sent表示句子 file表示文章名
            """
            line.append([c, ch['sent'], ch['line'], ch['doc'], ch['shengmu'], ch['yunmu'], ch['shengdiao'], ch['bihua'],
                         ch['cixing'], ch['sentiment'], ch['sentiment_score']])
    return line

# 进行整体计算
def name_overall_calc(name_dim):
    for ch in name_dim:
        td_idf = cal_tf_idf(ch)
        center_degree = cal_center_degree(ch)
        char_pos,pos = cal_pos(ch)
        ch.append(td_idf)
        ch.append(center_degree)
        ch.append(char_pos)
        ch.append(pos)
# 计算 tf_idf
def cal_tf_idf(ch):
    return 1
# 计算 中心度
# 简单来说，就是这个子在文章是否出于开头或者末尾
def cal_center_degree(ch):
    return 0
# 计算所在的char_pos, pos
def cal_pos(ch):
    char = ch[0]
    sent = ch[1]
    line = ch[2]
    sent_pos = line.index(sent)
    char_pos = sent.index(char)
    pos = sent_pos+char_pos
    return char_pos,pos

# 对某个字的联想
def hint_word(char):
    pass
def generate_idx():
    """
    读取诗经、易经、道德经的内容，并形成倒排索引
    :return:
    """
    widx = readFile('./诗经.txt')
    c_idx = read_character(widx,'鲍')
    name_dim = name_mapping(c_idx)
    name_overall_calc(name_dim)
    name_filter(name_dim)
    save_character_to_csv(name_dim, "./诗经.csv")

    # yijing_idx = readFile('/Users/danebrown/develop/nlp/易经.txt')
    # yijing_c_idx = read_character(yijing_idx)
    # save_character_to_csv(yijing_c_idx, '/Users/danebrown/develop/nlp/易经.csv')
    #
    # ddj_idx = readFile('/Users/danebrown/develop/nlp/道德经.txt')
    # ddj_c_idx = read_character(ddj_idx)
    # save_character_to_csv(ddj_c_idx, '/Users/danebrown/develop/nlp/道德经.csv')


def analyze():
    _init_stroke('/Users/danebrown/develop/nlp/strokes.txt')
    generate_idx()
    pass


if __name__ == "__main__":
    analyze()
