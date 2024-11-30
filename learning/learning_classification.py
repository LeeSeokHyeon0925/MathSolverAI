from processing import *
from network import *
import time
from Constant import *

# --- set gpu environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# --- model save
model_save_path = 'model/classification/'
saving = False
saving_point = 0
if not saving:
    saving_point = 0

# --- dataset load
print('Data Loading ...')
current_dir = os.path.dirname(os.path.abspath(__file__)) # load current path
math_dir = os.path.abspath(os.path.join(current_dir, '..'))
path = math_dir + '/problems/extended_problems/diff/'
    # train data consists of 7 classes with 10000 each samples // test data consists of 7 classes with 100 each samples
train_texts, train_cls = text_load(path + 'train/', cls_num * train_num)
test_texts, test_cls = text_load(path + 'test/', cls_num * test_num)
print('Loading Finished\n')

# --- build network
print('Network Building...')
model = NNv1(cls_num).to(DEVICE)
print('Build Finished\n')

loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
# epoch ~ 27.43
num_iter = 15000
batch_size = 128
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# --- learning
if saving:
    model.load_state_dict(torch.load(model_save_path + f'model_{saving_point}.pt'))
    model.eval()

start_time = time.time()

for it in range(saving_point, num_iter + 1):
    if it >= 10000:
        optimizer.param_groups[0]['lr'] = 0.0001
    # mini batch
    batch_text, batch_cls = mini_batch_training_text(train_texts, train_cls, batch_size)

    # training
    model.train()
    optimizer.zero_grad()
    prediction = model(torch.from_numpy(batch_text.astype(np.float32)).to(DEVICE))
    cls_tensor = torch.tensor(batch_cls, dtype=torch.long).to(DEVICE)

    train_loss = loss(prediction, cls_tensor)
    train_loss.backward()
    optimizer.step()

    # print loss
    if it % 1000 == 0:
        current_time = time.time() - start_time
        start_time = time.time()
        print(f'it : {it}   loss : {train_loss.item():.5f}  time: {current_time:.3f}')

    # save model and evaluation
    if it % 5000 == 0 and it != 0:
        print('Saving Model ...')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{it}.pt'))
        print('Saving Finished')

        model.eval()
        count = 0

        for i in range(test_num * cls_num):
            test_text = test_texts[i]

            with torch.no_grad():
                prediction = model(torch.from_numpy(test_text.astype(np.float32)).to(DEVICE))

            prediction = prediction.cpu().numpy()
            prediction = np.reshape(prediction, cls_num)
            prediction = np.argmax(prediction)

            get_cls = test_cls[i]

            if int(get_cls) == int(prediction):
                count += 1
        print(f'Accuracy: {count / (test_num * cls_num) * 100:.2f}\n')

'''
NNv1
Accuracy : 94.57
  Loss   :  0.32149

** error table **
[0 , 0 , 0 , 26, 0 , 0 , 0 ]
[0 , 0 , 0 , 12, 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]
[26, 12, 0 , 0 , 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]

---------------------------
NNv2
Accuracy : 92.71
  Loss   :  0.32223

** error table **
[0 , 0 , 0 , 26, 0 , 0 , 0 ]
[0 , 0 , 0 , 25, 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]
[26, 25, 0 , 0 , 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]
[0 , 0 , 0 , 0 , 0 , 0 , 0 ]

'''