import unicodedata
import string
import glob
import os
from tkinter import Toplevel, Button, Tk, Menu

import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tkinter import *

all_letters = string.ascii_letters + " .,;'"    #　abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'
n_letters = len(all_letters)        # 57
category_lines = {}
all_categories = []
criterion = nn.NLLLoss()
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


def unicodeToAscii(s):
    Ascii = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn' and c in all_letters:
            Ascii.append(c)
    return ''.join(Ascii)


def findFiles(path):
    return glob.glob(path)


def readLines(filename):
    lines = open(filename, 'r', encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def letterToIndex(letter):      # 找到letter在all_letters中的索引，例如"a" = 0, 'b' = 1
    return all_letters.find(letter)


def letterToTensor(letter):     # turn a letter into a <1 x n_letters> Tensor,'b' = tensor([[0., 1., 0., 0...
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):     # Turn a line into a <line_length x 1 x n_letters>
    tensor = torch.zeros(len(line), 1, n_letters)
    for index, letter in enumerate(line):
        tensor[index][0][letterToIndex(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def categoryFromOutput(output):
    category_i = output.data.topk(1)[1].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    """
    产生随机样本
    :return:
    """
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.Tensor([all_categories.index(category)]).long()
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s = s - m * 60
    return '%dm %ds ' % (m, s)


def train(category_tensor, line_tensor):
    hidden = model.initHidden()
    model.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


def evaluate(line_tensor):
    hidden = model.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        topv, topi = output.topk(n_predictions, 1, True)        # 获得top N的类别

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s ' % (value, all_categories[category_index]))
    return


path = '/Users/danebrown/develop/github/naming/data/'


def donothing():

   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()


def train():

    # 以下为训练

    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    current_loss = 0
    all_losses = []

    start = time.time()
    # 迭代次数
    for iter in range(1, n_iters + 1):
        # 生成随机样本
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 训练 核心
        output, loss = train(category_tensor, line_tensor)
        # 累计损失
        current_loss += loss
        # 每5000次 打印一次
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗(%s)' % category
            print('iter:{0} {1}% (time:{2}) loss:{3:.4f} {4} / {5} {6}'.format(iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        # 每1000 次打印一次
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            plt.plot(all_losses)
            current_loss = 0
    # 保存模型
    torch.save(model.state_dict(), path+'model.pth')
    plt.show()


def assumption1():
    # 以下为评估1
    model.load_state_dict(torch.load(path+'model.pth'))
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    for i in range(n_confusion):        # 通过一堆例子，记录哪些是正确的猜测
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    for i in range(n_categories):       # 将每一行除以其总和进行标准化处理
        confusion[i] = confusion[i] / confusion[i].sum()

    fig = plt.figure()      # 设置绘图
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + all_categories, rotation=90)      # 设置坐标轴
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))       # 在每一个刻度上强制贴上标签
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def assumption2():
    # 以下为评估2
    model.load_state_dict(torch.load(path+'model.pth'))
    n_prediction = 10000
    n_correct = 0
    for i in range(n_prediction):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        if guess == category:
            n_correct += 1
    print(n_correct / n_prediction)


def assumption3():
    model.load_state_dict(torch.load(path + 'model.pth'))
    predict('Dovesky')
    predict('Jackson')
    predict('Satoshi')

if __name__ == '__main__':

    for filename in findFiles(path + 'names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    n_hidden = 128
    model = RNN(n_letters, n_hidden, n_categories)  # 初始化

    root = Tk()
    menubar = Menu(root)
    menubar = Menu(root)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="train", command=train)
    filemenu.add_command(label="评估1", command=assumption1)
    filemenu.add_command(label="评估2", command=assumption2)
    filemenu.add_command(label="评估3", command=assumption3)


    filemenu.add_separator()

    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)
    editmenu = Menu(menubar, tearoff=0)
    editmenu.add_command(label="Undo", command=donothing)

    editmenu.add_separator()

    editmenu.add_command(label="Cut", command=donothing)
    editmenu.add_command(label="Copy", command=donothing)
    editmenu.add_command(label="Paste", command=donothing)
    editmenu.add_command(label="Delete", command=donothing)
    editmenu.add_command(label="Select All", command=donothing)

    menubar.add_cascade(label="Edit", menu=editmenu)
    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help Index", command=donothing)
    helpmenu.add_command(label="About...", command=donothing)
    menubar.add_cascade(label="Help", menu=helpmenu)

    root.config(menu=menubar)
    root.mainloop()



    # 以下为训练
    # criterion = nn.NLLLoss()
    # learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    #
    # n_iters = 100000
    # print_every = 5000
    # plot_every = 1000
    # current_loss = 0
    # all_losses = []
    #
    # start = time.time()
    # for iter in range(1, n_iters + 1):
    #     category, line, category_tensor, line_tensor = randomTrainingExample()
    #     output, loss = train(category_tensor, line_tensor)
    #     current_loss += loss
    #
    #     if iter % print_every == 0:
    #         guess, guess_i = categoryFromOutput(output)
    #         correct = '✓' if guess == category else '✗(%s)' % category
    #         print('iter:{0} {1}% (time:{2}) loss:{3:.4f} {4} / {5} {6}'.format(iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    #
    #     if iter % plot_every == 0:
    #         all_losses.append(current_loss / plot_every)
    #         plt.plot(all_losses)
    #         current_loss = 0
    #
    # torch.save(model.state_dict(), path+'model.pth')
    # plt.show()

    # 以下为评估1
    # model.load_state_dict(torch.load(path+'model.pth'))
    # confusion = torch.zeros(n_categories, n_categories)
    # n_confusion = 10000
    # for i in range(n_confusion):        # 通过一堆例子，记录哪些是正确的猜测
    #     category, line, category_tensor, line_tensor = randomTrainingExample()
    #     output = evaluate(line_tensor)
    #     guess, guess_i = categoryFromOutput(output)
    #     category_i = all_categories.index(category)
    #     confusion[category_i][guess_i] += 1
    #
    # for i in range(n_categories):       # 将每一行除以其总和进行标准化处理
    #     confusion[i] = confusion[i] / confusion[i].sum()
    #
    # fig = plt.figure()      # 设置绘图
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(confusion.numpy())
    # fig.colorbar(cax)
    #
    # ax.set_xticklabels([''] + all_categories, rotation=90)      # 设置坐标轴
    # ax.set_yticklabels([''] + all_categories)
    #
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))       # 在每一个刻度上强制贴上标签
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    #
    # plt.show()

    # 以下为评估2
    # model.load_state_dict(torch.load(path+'model.pth'))
    # n_prediction = 10000
    # n_correct = 0
    # for i in range(n_prediction):
    #     category, line, category_tensor, line_tensor = randomTrainingExample()
    #     output = evaluate(line_tensor)
    #     guess, guess_i = categoryFromOutput(output)
    #     if guess == category:
    #         n_correct += 1
    # print(n_correct / n_prediction)

    # 以下为评估3
    # model.load_state_dict(torch.load(path+'model.pth'))
    # predict('Dovesky')
    # predict('Jackson')
    # predict('Satoshi')
