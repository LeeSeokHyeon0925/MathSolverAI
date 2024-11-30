import cv2
import numpy as np
import pandas as pd
from Constant import *
from Learning.network import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'C:/Python/MathSolverAI/Learning/model/OCR/model_100000.pt'

def image_load(path, image_num):
    images = np.zeros((image_num, image_size[1], image_size[0]), dtype=np.uint8) # [num_image, h, w] = [700, 256, 512]
    texts = [''] * image_num # [700]

    cls_name = os.listdir(path)
    count = 0
    problem_count = image_num // len(cls_name) # number of problems

    for i in range(len(cls_name)):
        # load ground truth from xlsx file
        print(path + cls_name[i])
        texts_path = path + cls_name[i] + '/texts.xlsx'
        df = pd.read_excel(texts_path, engine='openpyxl')
        problem_data = df['problem']
        texts[count: count + problem_count] = problem_data

        # load image
        for j in range(problem_count):
            image_path = path + cls_name[i] + f'/{j}.jpg'
            image = cv2.imread(image_path)
            images[count, :, :] = image[:, :, 0]
            count += 1

    return images, texts

def preprocess_image(image):
    # image split by row
    count = 0

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 20), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=2)
    morph = cv2.erode(dilated, kernel, iterations=2)

    # text detection
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        can_go = True
        how = 3
        _, y, _, _ = cv2.boundingRect(contour)
        y -= how

        for i in range(len(boxes)):
            if boxes[i] - h // 2 < y < boxes[i] + h // 2:
                can_go = False; break

        if can_go:
            y = max(y, 0)
            boxes.append(y)

    boxes.sort()
    image_split = np.zeros((len(boxes), h, image_size[0]), dtype=np.uint8)
    # make new image
    for y in boxes:
        # cutting the next sentence
        horizontal_value = np.sum(binary_image[y:y + h, :], axis=1)
        end = len(horizontal_value) - 1
        for i in range(len(horizontal_value) - 1, word_size[1] // 2, -1):
            if horizontal_value[i] == 0:
                end = i + 1; break

        image_split[count, :end, :] = binary_image[y:y + end, :]
        show_image(cv2.bitwise_not(image_split[count, :, :]))
        count += 1

    return image_split

def image_segmentation(images):
    prediction = ''

    # place word in the center of 28 * 28
    def word_processor(image):
        horizontal_value = np.sum(image, axis=1)
        start = 0; end = word_size[1]

        for i in range(len(horizontal_value) // 2):
            if horizontal_value[i] != 0:
                start = i; break
        for i in range(len(horizontal_value) - 1, word_size[1] // 2, -1):
            if horizontal_value[i] != 0:
                end = i + 1; break

        image_temp = image[start:end, :]
        back_ground = np.zeros((word_size[1], word_size[0]), dtype=np.uint8)
        x_offset = (back_ground.shape[1] - image_temp.shape[1]) // 2
        y_offset = (back_ground.shape[0] - image_temp.shape[0]) // 2
        back_ground[y_offset:y_offset + image_temp.shape[0], x_offset:x_offset + image_temp.shape[1]] = image_temp

        return back_ground

    # cutting the word
    for image in images:
        vertical_value = np.sum(image, axis=0)
        char_regions = []
        start = 0
        in_char = False

        for i, value in enumerate(vertical_value):
            if value > 0 and not in_char:
                    start = i; in_char = True
            elif value == 0 and in_char:
                    char_regions.append((start, i)); in_char = False

        # save words as single and double segmentation
        image_segm_single = np.zeros((len(char_regions), word_size[1], word_size[0]),dtype=np.uint8)
        image_segm_double = np.zeros((len(char_regions) - 1, word_size[1], word_size[0]),dtype=np.uint8)

        for i, (start, end) in enumerate(char_regions):
            image_temp = image[:, start:end]
            if image_temp.shape[1] > word_size[0] - 3:
                ratio = (word_size[0] - 3) / image_temp.shape[1]
                image_temp = cv2.resize(image_temp, (int(image_temp.shape[0] * ratio), int(image_temp.shape[1] * ratio)))
            image_segm_single[i, :, :] = word_processor(image_temp)
            show_image(cv2.bitwise_not(image_segm_single[i, :, :]))

        for i in range(len(char_regions) - 1):
            if mode == 1:
                image_temp = image[:, char_regions[i][0]:char_regions[i + 1][1]]
            elif mode == 2:
                image_temp = image[:, char_regions[i][0]:char_regions[i][0] + word_size[0]]

            if image_temp.shape[1] > word_size[0] - 3:
                ratio = (word_size[0] - 3) / image_temp.shape[1]
                image_temp = cv2.resize(image_temp, (int(image_temp.shape[0] * ratio), int(image_temp.shape[1] * ratio)))
            image_segm_double[i, :, :] = word_processor(image_temp)
            show_image(cv2.bitwise_not(image_segm_double[i, :, :]))

        prediction += image_ocr(image_segm_single, image_segm_double)

    return prediction

def image_ocr(image_segm_single, image_segm_double):

    # load model
    model = CNN(sum(num for num in word_count)).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    def evaluation(image):
        input_data = np.zeros((1, 1, word_size[1], word_size[0]), dtype=np.float32)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        input_data[0, 0, :, :] = binary_image / 255.0

        with torch.no_grad():
            output_data = model(torch.from_numpy(input_data.astype(np.float32)).to(DEVICE))

        output_data = output_data.cpu().numpy()
        output_data = np.reshape(output_data, sum(num for num in word_count))
        output_data = torch.tensor(output_data)
        output_data = torch.nn.functional.softmax(output_data, dim=0)

        return output_data

    result = []

    stack = [i for i in range(len(image_segm_single) -1, -1, -1)]
    while stack:
        i = stack.pop()
        output_single = evaluation(image_segm_single[i, :, :])
        if stack:
            output_double = evaluation(image_segm_double[i, :, :])
        else:
            output_double = [0, 0]

        if max(output_double) > 0.8:
            result.append(np.argmax(output_double)); stack.pop()
        else:
            result.append(np.argmax(output_single))

    # convert to sentence
    prediction = ''
    f = open(f'./word_gt.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    for it in result:
        prediction += lines[it][:-1]

    return prediction

def show_image(image):
    if render:
        cv2.imshow('image', image)
        cv2.waitKey(0)

from Levenshtein import distance as levenshtein_distance

def calculate_percentage(str1, str2):
    distance = levenshtein_distance(str1, str2)

    similarity = round((1 - (distance / max(len(str1), len(str2)))) * 100, 2)
    print(f"    similarity    : {similarity} %")
    return similarity

render = False # view the image
mode = 1

percentage = []
if __name__ == '__main__':
    images, texts = image_load('C:/Python/MathSolverAI/problems/extended_problems/diff/test/', test_num * cls_num)
    for (image, text) in zip(images, texts):
        show_image(image)
        image_split = preprocess_image(image)
        prediction = image_segmentation(image_split)
        print('original sentence :', text.replace(' ', ''))
        print('  OCR    sentence :', prediction)
        percentage.append(calculate_percentage(text.replace(' ', ''), prediction))
    print(f' total similarity : {sum(percentage) / (test_num * cls_num):.2f} %')
    print(f'   Max Accuracy   : {max(percentage)} %')
    print(f'   Min Accuracy   : {min(percentage)} %')
'''
mode
    if mode == 1:
        image_temp = image[:, char_regions[i][0]:char_regions[i + 1][1]]
    elif mode == 2:
        image_temp = image[:, char_regions[i][0]:char_regions[i][0] + word_size[0]]
    
----------------------------------------
mode_1
  Accuracy   : 30.13 %
Max Accuracy : 59.46 %
Min Accuracy :  1.92 %


mode_2
  Accuracy   : 24.74 %
Max Accuracy : 51.35 %
Min Accuracy :  3.23 %

'''
