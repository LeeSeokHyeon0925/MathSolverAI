from Learning.network import *
from Learning.processing import tokenizer_korean
import numpy as np
from Constant import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'C:/Python/MathSolverAI/Learning/model/classification/model_15000.pt'

def classification_problem(sentence):
    model = NNv1(cls_num).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    token = tokenizer_korean(sentence)
    input_data = np.zeros((1, max_length))
    input_data[0] = token / token_norm

    with torch.no_grad():
        outputs = model(torch.from_numpy(input_data.astype(np.float32)).to(DEVICE))

    outputs = outputs.cpu().numpy()
    outputs = np.reshape(outputs, cls_num)
    outputs = np.argmax(outputs)

    return outputs