# coding=utf-8
import re
import torch
import torch.nn as nn
from zhon.hanzi import punctuation
import jieba
import jieba.analyse
import jieba.posseg
import jiagu
import logging
import synonyms

import os
from torch.autograd import Variable
import pandas as pd
from pypinyin import pinyin, lazy_pinyin, Style
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

vectorizer = CountVectorizer()
transformer = TfidfTransformer()


"""
要跳过的词
"""
skip_word = punctuation + '□'
pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| ' \
          r'|…|（|）|，|。|；|：|？|（|）|【|】|「|」|！|、|\||《|》|\" |\"'
strokes = []
# 可以保留的韵母
yunmuStay = ['a', 'o', 'e', 'eu', 'wu', 'yu']
# 要跳过的声母，因为名字有爆破音，因此不需要再次出现爆破音
skip_shengmu = ['b', 'p', 't', 'd', 'k']
# 要跳过的词性，名词、代词、动词、量词都要跳过。因为这些词在名字中不好听
skip_cixing = ['n', 'r', 'v', 'm', 'p', 'd', 'z']
bad_name=["人","女","下","反","凶","尸","勿","犯","违","讼","丧","灭","媾","刑","宰","灾","乱","匪","穷","梏","贱","吝","患","害","笨","病","亡","丑"]

def _cut(line):
    """
    断句
    :param line:
    :return:
    """
    split = re.split(pattern, line)
    return split


def is_Chinese(ch):
    """
    判断是否为中文
    :param ch: 字符串
    :return: 是否为中文
    """
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False


# 读文件
def readFile(fileName):
    """
    读文件
    :param fileName: 文件名称
    :return:数据格式
    word_idx{
        word(切割出来的词):{
            line:'一行内容',
            file:'文件名'
            hash:'hash值'
        }
    }
    """
    word_idx = {}
    """
    :param fileName:
    seg_list:一句话
    line 行
    file 文章
    :return: 语句,行,文章名称的map
    """

    f = open(fileName)
    inserted_hash_tag = []
    for line in f.readlines():
        if line in skip_word or line.isdecimal() or line.isdigit() or line.isspace():
            continue

        # seg_list = jieba.cut(line, use_paddle=True)
        seg_list = _cut(line)  # 切割成一句话
        # 指的是哪篇文章的哪句话
        # word表示的是切割出来的词
        # line表示一行 fileName表示文章名称
        hash = str(line.__hash__()) + '_' + str(fileName.__hash__())
        doc = {'line': line, 'file': fileName, 'hash': hash}
        for word in seg_list:
            if word in skip_word or word.isdecimal() or word.isdigit() or word.isspace():
                continue
            if word in bad_name:
                continue
            if word not in word_idx.keys():
                word_idx[word] = []
            """倒排索引"""
            logging.debug("解析出来的seg_list的每一条信息是:%s", word)
            # 去重
            docs = [d for d in word_idx.get(word) if d['hash'] == hash]
            if docs is None or len(docs) == 0:
                word_idx.get(word).append(doc)
    return word_idx


def gen_ShengDiao(char):
    """
    计算声调
    :param char:文字
    :return:声调
    """
    shengdiao = 0
    if pinyin(char, style=Style.TONE2) \
            and pinyin(char, style=Style.TONE2)[0][0] \
            and re.findall(r'\d+', pinyin(char, style=Style.TONE2)[0][0]):
        shengdiao = re.findall(r'\d+', pinyin(char, style=Style.TONE2)[0][0])[0]
    return shengdiao


def gen_cixing(char):
    """
    计算词性，名词、动词、形容词等
    :param char:文字
    :return:词性
    """
    wordCut = jieba.posseg.cut(char)
    # 词性
    cixing = 'n'
    for w, f in wordCut:
        cixing = f
    return cixing


# 生成索引
def read_character(W_IDX):
    """
    生成名字属性矩阵
    :param W_IDX: 读进来的文件
    :return:
    """
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
            # 词性
            cixing = gen_cixing(char)

            # 如果是空格、数字等等，重来
            if char in skip_word or char.isdecimal() or char.isdigit() or char.isspace():
                continue
            # 初始化
            if char not in character_idx.keys():
                character_idx[char] = []
            # 这里做相似性处理

            doc_idx = W_IDX.get(word)
            # 拼音，声调，笔画数
            shengmu = pinyin(char, style=Style.INITIALS)[0][0]
            yunmu = pinyin(char, style=Style.FINALS)[0][0]

            # 声调
            shengdiao = gen_ShengDiao(char)

            # 笔画
            bihua = get_stroke(char)

            for w in doc_idx:
                # sentiment positive 还是negivate。以及对应的分数
                # 这里看的是切割出来的每一个单句，的情感分析。因为一个字可能出现在多个文章中，但是由于正片文章情感波动太大，难以用
                # 这个作为参考。但是又不能直接用 字 本身作为情感分析，因为情感分析要求有上下文，上下文越多，越能体现情感。因此这里折中
                # 使用切分出来的字句
                sentiment = jiagu.sentiment(word.strip())
                char_sentiment = jiagu.sentiment(char)
                # 这里要计算字符的信息
                # 建议规则 笔画要少，
                # 词性要求必须是好的
                # 不要动名量词
                # 计算每个字在每句话中的position
                # 计算每个字在每句话中的TF-IDF
                node = {'sent': word.strip(),
                        'line': w['line'],
                        'doc': w['file'],
                        'shengmu': shengmu,
                        'yunmu': yunmu,
                        'shengdiao': shengdiao,
                        'bihua': bihua,
                        'cixing': cixing,
                        'char_sentiment': -char_sentiment[1] if char_sentiment[0] == 'negative' else char_sentiment[1],
                        'sentiment_score': -sentiment[1] if sentiment[0] == 'negative' else sentiment[1],
                        'word_len': len(word.strip())
                        }
                character_idx.get(char).append(node)
    return character_idx


def _init_stroke(strokes_path):
    """
    初始化笔画工具
    :param strokes_path:
    :return:
    """
    with open(strokes_path, 'r') as fr:
        for line in fr:
            strokes.append(int(line.strip()))


def get_stroke(c):
    """
    计算笔画
    :param c:中文字
    :return: 笔画
    """
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段
    unicode_ = ord(c)
    if 13312 <= unicode_ <= 64045:
        return strokes[unicode_ - 13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes[unicode_ - 80338]
    else:
        -1
        # can also return 0


# 保存到csv
def save_character_to_csv(line, file_name):
    """
    保存到CSV
    :param line: 要保存的内容
    :param file_name: 文件名
    :return:
    """

    columns = ['character', 'sentence', 'line', 'document', 'shengmu',
               'yunmu', 'shengdiao', 'bihua', 'cixing','char_sentiment',
               'sentiment_score', 'sentence_length', 'td_idf', 'degree', 'char_pos',
               'pos','is_begin_of_sent','is_end_of_sent','is_begin_of_line','is_end_of_line']
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
def name_filter(name_dim, family_name, skip_level=100):
    """
    根据一些条件进行过滤
    :param name_dim:
    :param family_name:
    :param skip_level:
    :return:
    """
    """
    :parameter name_dim 要处理的对象数组
    :parameter family_name 姓
    :parameter skip_level 过滤等级，越低过滤的越少，越高过滤的越多。
    1级:过滤第一个字中的声母、韵母
    2级：过滤掉指定的声母
    3级：过滤掉3，4声
    4级：2声，而且不是保留韵母的，则过滤掉
    """
    first_name_shengmu = pinyin(family_name, style=Style.INITIALS)[0][0]
    first_name_yunmu = pinyin(family_name, style=Style.FINALS)[0][0]
    result = []
    for ch in name_dim:
        char = ch[0]
        sent = ch[1]
        line = ch[2]
        doc = ch[3]
        shengmu = ch[4]
        yunmu = ch[5]
        shengdiao = ch[6]
        bihua = ch[7]
        cixing = ch[8]
        sentiment = ch[9]
        sentiment_score = ch[10]
        sentence_length = ch[11]
        td_idf=ch[12]
        degree=ch[13]
        char_pos=ch[14]
        pos=ch[15]
        is_begin_of_sent=ch[16]
        is_end_of_sent=ch[17]
        is_begin_of_line=ch[18]
        is_end_of_line=ch[19]

        if (shengmu == first_name_shengmu or yunmu == first_name_yunmu) and skip_level > 1:
            continue
        # if len([c for c in skip_cixing if c in cixing]) > 0:
        #     continue
        # if (cixing in skip_cixing):
        #     continue
        if shengmu in skip_shengmu and skip_level > 2:
            continue
        if shengdiao not in ['1', '2','3','4'] and skip_level > 3:
            continue
            # 如果声调是2声，而且韵母不是a,o,e,i,wu,yu，则跳过
        if (shengdiao == 2 and yunmu not in yunmuStay) and skip_level > 4:
            continue
        if not is_end_of_line \
                and not is_begin_of_line \
                and not is_begin_of_sent \
                and not is_end_of_sent \
                and skip_level > 5:
            continue
        # if bihua > 12:
        #     continue
        # if sentiment == 'negative' or sentiment_score < 0.75:
        #     continue
        result.append(ch)

    return result


# 转成N维数组
def name_mapping(c_idx):
    """
    将字的对象数组转成数组
    :param c_idx:
    :return:
    """
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
                         ch['cixing'],
                         ch['char_sentiment'],
                         ch['sentiment_score'], ch['word_len']])
    return line


# 进行整体计算
def name_overall_calc(name_dim):
    """
    进行一些字的属性扩展，例如td_idf，度中心等等
    :param name_dim: 名字属性表
    :return:
    """
    for ch in name_dim:
        td_idf = 1
        center_degree = cal_center_degree(ch)
        char_pos, pos,is_begin_of_sent,is_end_of_sent,is_begin_of_line,is_end_of_line = cal_pos(ch)
        ch.append(td_idf)
        ch.append(center_degree)
        ch.append(char_pos)
        ch.append(pos)
        ch.append(is_begin_of_sent)
        ch.append(is_end_of_sent)
        ch.append(is_begin_of_line)
        ch.append(is_end_of_line)


# 计算 tf_idf
def cal_tf_idf(line):
    """
    计算tf_idf
    :param line:
    :return:
    """
    X = vectorizer.fit_transform([line])
    word = vectorizer.get_feature_names()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()

    return 1


# 计算 中心度
# 简单来说，就是这个字在文章是否出于开头或者末尾
def cal_center_degree(ch):
    """
    计算读中心
    :param ch:
    :return:
    """
    # 越靠近两边，这个度越高，越靠近中心度越低
    char = ch[0]
    sent = ch[1]
    line = ch[2]
    return 0


# 计算所在的char_pos, pos
def cal_pos(ch):
    """
    计算字符所在位置
    :param ch:
    char 是字
    sent 是句子
    line 是文章
    :return:
    """
    char = ch[0]
    sent = ch[1]
    line = ch[2]
    sent_pos = line.index(sent)
    char_pos = sent.index(char)
    pos = sent_pos + char_pos
    char_idx_of_line = line.index(char)
    is_begin_of_sent = char_pos == 0
    is_end_of_sent = char_pos == len(sent)
    is_begin_of_line = char_idx_of_line == 0
    is_end_of_line = char_idx_of_line == len(line)

    return char_pos, pos,is_begin_of_sent,is_end_of_sent,is_begin_of_line,is_end_of_line


# 对某个字的联想
def hint_word(name_dim):
    """
    对字进行相似字联想，不建议开启，性能很差
    :param name_dim:
    :return:
    """
    hints = []
    # 已经处理过的话，不再处理
    processed = []
    for ch in name_dim:
        char = ch[0]
        # 处理过的字符串就不再处理
        if len([p for p in processed if p == char]) > 0:
            continue
        processed.append(char)
        hintChar = synonyms.nearby(char)[0][1:3]
        if len(hintChar) > 0:
            for h in hintChar:
                for w in h:
                    if is_Chinese(w):
                        if len([p for p in hints if p[0] == w]) <= 0:
                            newChar = w
                            sent = ''
                            line = ''
                            doc = ''
                            shengmu = pinyin(newChar, style=Style.INITIALS)[0][0]
                            yunmu = pinyin(newChar, style=Style.FINALS)[0][0]
                            # 声调
                            shengdiao = gen_ShengDiao(newChar)
                            # 笔画
                            bihua = get_stroke(newChar)
                            cixing = gen_cixing(newChar)
                            sentiment_tmp = jiagu.sentiment(newChar)

                            char_sentiment = -sentiment_tmp[1] if sentiment_tmp[0] == 'negative' else sentiment_tmp[1]
                            sentiment_score = 0
                            hints.append([newChar, sent, line, doc, shengmu, yunmu, shengdiao, bihua,
                                          cixing,
                                          char_sentiment,
                                          sentiment_score])

    return hints

def generate_idx():
    """
    读取诗经、易经、道德经的内容，并形成倒排索引，数据预处理
    :return:
    """

    widx = readFile('./唐诗.txt')
    c_idx = read_character(widx)
    name_dim = name_mapping(c_idx)
    name_overall_calc(name_dim)
    # hintWord = hint_word(name_dim)
    # name_dim.extend(hintWord)
    result = name_filter(name_dim, '鲍', -1)

    save_character_to_csv(result, "./唐诗.csv")

    # yijing_idx = readFile('/Users/danebrown/develop/nlp/易经.txt')
    # yijing_c_idx = read_character(yijing_idx)
    # save_character_to_csv(yijing_c_idx, '/Users/danebrown/develop/nlp/易经.csv')
    #
    # ddj_idx = readFile('/Users/danebrown/develop/nlp/道德经.txt')
    # ddj_c_idx = read_character(ddj_idx)
    # save_character_to_csv(ddj_c_idx, '/Users/danebrown/develop/nlp/道德经.csv')


def analyze():
    #加载声调信息
    _init_stroke('/Users/danebrown/develop/nlp/strokes.txt')
    generate_idx()

    pass


if __name__ == "__main__":
    analyze()
