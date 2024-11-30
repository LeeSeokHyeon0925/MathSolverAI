from Learning.network import *
import numpy as np
from Constant import *
import cv2

# --- set the GPU environment
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- load OCR model
model_path = '/Learning/model/OCR/model_100000.pt'
model = CNN(sum(num for num in word_count)).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# --- test
count = 0

for i in range(sum(num for num in word_count)):
    test_path = f'./word/test/{i}.jpg'

    image_ori = cv2.imread(test_path)
    image_temp = cv2.bitwise_not(image_ori)
    image = np.zeros((1, 1, word_size[1], word_size[0]), dtype=np.float32)

    image[0, 0, :, :] = image_temp[:, :, 0]
    image /= 255.0

    with torch.no_grad():
        output_data = model(torch.from_numpy(image.astype(np.float32)).to(DEVICE))

    output_data = output_data.cpu().numpy()
    output_data = np.reshape(output_data, sum(num for num in word_count))
    output_data = torch.tensor(output_data)
    output_data = torch.nn.functional.softmax(output_data, dim=0)

    f = open(f'./word_gt.txt', 'r', encoding='utf-8')

    lines = f.readlines()
    prediction = lines[np.argmax(output_data)][:-1]
    if prediction == lines[i][:-1]:
        count += 1
    else:
        print('original sentence : ', lines[i][:-1], '  OCR sentence : ', prediction)
        print('------------------------------------------------')

print(count / sum(num for num in word_count) * 100, '%')