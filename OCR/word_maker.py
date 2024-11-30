import os
import matplotlib.pyplot as plt, pandas as pd
from matplotlib import font_manager as fm
from PIL import Image
import random, io, zipfile
from Constant import *

# --- basic setting
text_color = (0, 0, 0) # black color
bg_color = (255, 255, 255) # white color
font_list = ['Batang.ttc', 'Gulim.ttc', 'NanumGothic.ttf', 'NGULIM.ttf', 'HMFMOLD.ttf']

# --- load word
cls = []
f = open(f'./word.txt', 'r', encoding='utf-8')
lines = f.readlines()
    # load ground truth from txt file
for it in range(len(lines)):
    cls.append((lines[it])[:-1])
# --- making
count = 0

for char in cls:
    path = f'./word/train/'
    if not os.path.exists(path):
        os.makedirs(path)

    for _ in range(word_num):
        image_path = f'{path}/{count}.jpg'
        count += 1
        font_path = f'C:/Windows/Fonts/{random.choice(font_list)}'
        font_size = round(random.uniform(10, 12), 1)
        rotation_angle = random.uniform(-10, 10)

        problem_text = str(char)

        prop = fm.FontProperties(fname=font_path)
        plt.figure(figsize=(word_size[0]/100, word_size[1]/100))
        plt.text(0.5, 0.5, problem_text, fontsize=font_size, ha='center', va='center', color=text_color, fontproperties=prop)
        plt.axis('off')

        with io.BytesIO() as buf:
            plt.savefig(buf, format='jpg')
            buf.seek(0)

            image = Image.open(buf)
            image = image.rotate(rotation_angle, expand=False, fillcolor=bg_color)
            image = image.convert('L')
            image.save(image_path)

        plt.close()
    print(f'class-{char} completed')
print('train completed')

count = 0

for char in cls:
    path = f'./word/test/'
    if not os.path.exists(path):
        os.makedirs(path)

    image_path = f'{path}/{count}.jpg'
    count += 1
    font_path = f'C:/Windows/Fonts/{random.choice(font_list)}'
    font_size = round(random.uniform(10, 12), 1)
    rotation_angle = random.uniform(-10, 10)

    problem_text = str(char)

    prop = fm.FontProperties(fname=font_path)
    plt.figure(figsize=(word_size[0]/100, word_size[1]/100))
    plt.text(0.5, 0.5, problem_text, fontsize=font_size, ha='center', va='center', color=text_color, fontproperties=prop)
    plt.axis('off')

    with io.BytesIO() as buf:
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        image = Image.open(buf)
        image = image.rotate(rotation_angle, expand=False, fillcolor=bg_color)
        image = image.convert('L')
        image.save(image_path)

    plt.close()
print('test completed')