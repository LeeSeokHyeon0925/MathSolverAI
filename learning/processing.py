import os, cv2

import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from Constant import *

# string tokenization with transformers library
def tokenizer_korean(texts):
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    token = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt',
                      truncation_strategy='longest_first')
    token_ids = token['input_ids'].numpy()
    return token_ids

# load text data from .xlsx
def text_load(path, text_num):
    texts = np.zeros((text_num, max_length)) # [num_text, max_length] = [70,000, 64]
    cls = np.zeros(text_num, dtype=np.uint8) # [70,000]

    cls_name = os.listdir(path)
    count = 0
    problem_count = text_num // len(cls_name) # number of problems

    for i in range(len(cls_name)):
        texts_path = path + cls_name[i] + '/texts.xlsx'
        print(texts_path)
        df = pd.read_excel(texts_path, engine='openpyxl')
        problem_data = df['problem']

        token_ids = tokenizer_korean(problem_data.tolist())
        texts[count:count + problem_count] = token_ids
        cls[count:count + problem_count] = i
        count += problem_count

    return texts, cls

# mini batch
def mini_batch_training_text(train_text, train_cls, batch_size):
    batch_text = np.zeros((batch_size, max_length)) # [b, 64]
    batch_cls = np.zeros(batch_size) # [b]

    rand_num = np.random.randint(0, train_text.shape[0], size=batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        batch_text[it, :] = train_text[temp, :] / token_norm # normalization
        batch_cls[it] = train_cls[temp]

    return batch_text, batch_cls

# mini batch of zip file
def mini_batch_training_zip(z_file, z_file_list, train_cls, batch_size):
    batch_image = np.zeros((batch_size, word_size[1], word_size[0])) # [b, 28, 28]
    batch_cls = np.zeros(batch_size) # [b]

    rand_num = np.random.randint(0, len(z_file_list), size=batch_size)
    for it in range(batch_size):
        temp = rand_num[it]
        image_temp = z_file.read(z_file_list[temp])
        image_temp = cv2.imdecode(np.frombuffer(image_temp, np.uint8), 1)
        # binary_image = cv2.bitwise_not(image_temp) # model 1
        _, binary_image = cv2.threshold(image_temp, 127, 255, cv2.THRESH_BINARY_INV) # model 2
        image_temp = binary_image.astype(np.float32)

        batch_image[it, :, :] = image_temp[:, :, 0] / 255.0
        batch_cls[it] = train_cls[temp]

    return batch_image, batch_cls