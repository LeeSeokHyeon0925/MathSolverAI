import matplotlib.pyplot as plt, pandas as pd
from matplotlib import font_manager as fm
from PIL import Image

import random, problems, io, os
from Constant import *

# --- basic setting
filename = 'diff' # problem category filename
text_color = (0, 0, 0) # black color
bg_color = (255, 255, 255) # white color
output_path_basic = f'./extended_problems/' # path of basic
font_list = ['Batang.ttc', 'Gulim.ttc', 'NanumGothic.ttf', 'NGULIM.ttf', 'HMFMOLD.ttf']

# --- making
for cls in range(cls_num):
    data = {'problem': []}
    path = f'{output_path_basic}/{filename}/train/{cls}'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(train_num):
        problem_text = problems.maker(cls)

        data['problem'].append(
            problem_text.replace('$', '').replace('{', '').replace('}', '').replace('\n', '').replace('\\', ''))

    df = pd.DataFrame(data)
    df.to_excel(f'{path}/texts.xlsx', index=False, engine='openpyxl')
    print(f'class-{cls} completed')
print('train completed')

for cls in range(cls_num):
    data = {'problem': []}
    path = f'{output_path_basic}/{filename}/test/{cls}'

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(test_num):
        image_path = f'{path}/{i}.jpg'
        font_path = f'C:/Windows/Fonts/{random.choice(font_list)}'
        font_size = round(random.uniform(15, 17), 1)

        problem_text = problems.maker(cls)

        prop = fm.FontProperties(fname=font_path)
        plt.figure(figsize=(image_size[0]/100, image_size[1]/100))
        plt.text(0.5, 0.5, problem_text, fontsize=font_size, ha='center', va='center', color=text_color, fontproperties=prop)
        plt.axis('off')

        with io.BytesIO() as buf:
            plt.savefig(buf, format='jpg')
            buf.seek(0)

            image = Image.open(buf)
            image = image.convert('L')
            image.save(image_path)

        plt.close()

        path_excel = f'{path}/texts.xlsx'
        data['problem'].append(
            problem_text.replace('$', '').replace('{', '').replace('}', '').replace('\n', '').replace('\\', ''))

    df = pd.DataFrame(data)
    df.to_excel(path_excel, index=False, engine='openpyxl')
    print(f'class-{cls} completed')
print('test completed')
