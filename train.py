import argparse
import os
import re
import numpy as np
from collections import Counter
import pickle


class NGramModel:
    def __init__(self):
        # В каждом словаре префикс служит индексом и для каждого префикса хранится Counter, содержащий
        # следующее возможное слово и сколько раз оно использовалось после данного префикса
        self.bigram = dict()  # использует 1 предыдущее слово
        self.trigram = dict()  # использует 2 предыдущих слова
        self.ngram = dict()  # использует 3 предыдущих слова

    def fit(self, directory):
        # directory - строковая переменная, содержит название папки с текстами для обучения

        if directory:  # если папка дана
            files_in_directory = os.listdir(os.path.abspath(os.curdir) + '/' + directory)  # список имен файлов
            for file in files_in_directory:
                with open(args.input_dir + '/' + file, encoding='utf-8') as f:
                    # обрабатываем и получаем список слов из букв и цифр
                    text = re.sub("\W+", " ", " ".join(f.readlines())).lower().split()
                    self.process_text(text)
        else:  # Считываем текст из stdin. Используем стоп-строку "EndOfText"
            print("Input text (stop with line 'EndOfText'):")
            line = ""  # текущая считанная строка
            text = []  # список строк

            while line != "EndOfText":
                text.append(line)
                line = input()

            # получаем список слов
            text = re.sub("\W+", " ", " ".join(text)).lower().split()
            self.process_text(text)

    # метод, чтобы обрабатывать тексты из файлов и тексты из стандартного входного потока одинаковым образом
    def process_text(self, text):
        # Храним последние слова (префиксы очередного слова), чтобы использовать как индексы
        bi_prefix = text[0]
        tri_prefix = [None, text[0]]
        n_prefix = [None, None, text[0]]

        # В реализации так оказалось проще, но затратнее по памяти. В модели етсь над чем работать

        # Для текста сохраняем все возможные n-граммы в локальные словари
        # и затем складываем все в общие словари
        local_bigram = dict()
        local_trigram = dict()
        local_ngram = dict()

        for i in range(1, len(text)):
            if bi_prefix not in local_bigram.keys():
                local_bigram[bi_prefix] = [text[i]]
            else:
                local_bigram[bi_prefix].append(text[i])
            bi_prefix = text[i]

            if tri_prefix[0] is not None:
                if (tri_prefix[0], tri_prefix[1]) not in local_trigram.keys():
                    local_trigram[tuple(tri_prefix)] = [text[i]]
                else:
                    local_trigram[tuple(tri_prefix)].append(text[i])
            tri_prefix[1:].append(text[i])

            if n_prefix[0] is not None:
                if (n_prefix[0], n_prefix[1], n_prefix[2]) not in local_ngram.keys():
                    local_ngram[tuple(n_prefix)] = [text[i]]
                else:
                    local_ngram[tuple(n_prefix)].append(text[i])
            n_prefix[1:].append(text[i])

        for bi_prefix in local_bigram.keys():
            if bi_prefix not in self.bigram.keys():
                self.bigram[bi_prefix] = Counter(local_bigram[bi_prefix])
            else:
                self.bigram[bi_prefix] += Counter(local_bigram[bi_prefix])

        for tri_prefix in local_trigram.keys():
            if tri_prefix not in self.trigram.keys():
                self.trigram[tuple(tri_prefix)] = Counter(local_trigram[tuple(tri_prefix)])
            else:
                self.trigram[tuple(tri_prefix)] += Counter(local_trigram[tuple(tri_prefix)])

        for n_prefix in local_ngram.keys():
            if n_prefix not in self.ngram.keys():
                self.ngram[tuple(n_prefix)] = Counter(local_ngram[tuple(n_prefix)])
            else:
                self.ngram[tuple(n_prefix)] += Counter(local_ngram[tuple(n_prefix)])

    def generate(self, prefix=None, length=1):
        # Если префикс не дан, выбираем случайный
        if prefix is None:
            prefix = np.random.choice(list(self.bigram.keys()))

        # generated - строковая переменная хранит сгенерированную строку
        generated = prefix

        # Используем такие же префиксы
        bi_prefix = None
        tri_prefix = [None, None]
        n_prefix = [None, None, None]

        # Извлекаем из введенного префикса слова, чтобы получить префикса для генерирования следующих слов
        for i in prefix.split():
            bi_prefix = i
            tri_prefix = tri_prefix[1:] + [i]
            n_prefix = n_prefix[1:] + [i]

        TOP_N_WORDS = 30
        for i in range(length):
            choice_list = []  # список возможных слов, содержит по 10 самых популярных продолжений каждой модели
            if bi_prefix in self.bigram.keys():
                choice_list.append(self.bigram[bi_prefix].most_common(TOP_N_WORDS))
            if tuple(tri_prefix) in self.trigram.keys():
                choice_list.append(self.trigram[tuple(tri_prefix)].most_common(TOP_N_WORDS))
            if tuple(n_prefix) in self.ngram.keys():
                choice_list.append(self.ngram[tuple(n_prefix)].most_common(TOP_N_WORDS))
            if not choice_list:
                choice_list.append(np.random.choice(list(self.bigram.keys())))

            # Выбираем случайное слово из самых популярных
            new_word = choice_list[0][np.random.choice(len(*choice_list)) - 1][0]

            # Обновляем префиксы
            bi_prefix = new_word
            tri_prefix = tri_prefix[1:] + [new_word]
            n_prefix = n_prefix[1:] + [new_word]

            # Добавляем новое слово в итоговую строку
            generated += " " + new_word

        return generated


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()

args.input_dir = 'data'
args.model = "model.pkl"

model = NGramModel()
model.fit(args.input_dir)

with open(args.model, "wb") as file:
    pickle.dump(model, file)
