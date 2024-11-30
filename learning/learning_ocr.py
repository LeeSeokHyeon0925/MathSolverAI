from processing import *
from network import *
import time, zipfile
from Constant import *

# --- set gpu environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# --- model save
model_save_path = 'model/OCR/'
saving = True
saving_point = 20000
if not saving:
    saving_point = 0

# --- dataset load
print('Loading Zip ...')
current_dir = os.path.dirname(os.path.abspath(__file__)) # load current path
math_dir = os.path.abspath(os.path.join(current_dir, '..'))
path = math_dir + '/OCR/word/'

z_train = zipfile.ZipFile(path + 'train.zip', 'r')
z_train_list = z_train.namelist()[1:]
# sort (0, 1, 10, ...) -> (0, 1, 2, ...)
z_train_list = sorted(z_train_list, key=lambda x: int(x.split('/')[1].split('.')[0]))
train_cls = [i for i in range(sum(num for num in word_count)) for _ in range(word_num)]

z_test = zipfile.ZipFile(path + 'test.zip', 'r')
z_test_list = z_test.namelist()[1:]
# sort (0, 1, 10, ...) -> (0, 1, 2, ...)
z_test_list = sorted(z_test_list, key=lambda x: int(x.split('/')[1].split('.')[0]))
test_cls = [i for i in range(sum(num for num in word_count))]
print('Loading Finished\n')

# --- build network
print('Network Building...')
model = CNN(sum(num for num in word_count)).to(DEVICE)
print('Build Finished\n')

loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
# epoch ~ 26.2
num_iter = 100000
batch_size = 64
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate
                            , weight_decay=5e-4, momentum=0.9)

# --- learning
if saving:
    model.load_state_dict(torch.load(model_save_path + f'model_{saving_point}.pt'))
    model.eval()

start_time = time.time()

for it in range(saving_point, num_iter + 1):
    if 80000 < it <= 90000:
        optimizer.param_groups[0]['lr'] = 0.001
    elif 90000 < it <= 100000:
        optimizer.param_groups[0]['lr'] = 0.0001
    # mini batch
    batch_image, batch_cls = mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size)
    batch_image = np.expand_dims(batch_image, axis=1)

    # training
    model.train()
    optimizer.zero_grad()
    prediction = model(torch.from_numpy(batch_image.astype(np.float32)).to(DEVICE))
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
    if it % 10000 == 0 and it != 0:
        print('Saving Model ...')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model.state_dict(), os.path.join(model_save_path, f'model_{it}.pt'))
        print('Saving Finished')

        model.eval()
        count = 0
        for i in range(sum(num for num in word_count)):
            image_temp = z_test.read(z_test_list[i])
            image_temp = cv2.imdecode(np.frombuffer(image_temp, np.uint8), 1)
            image_temp = cv2.bitwise_not(image_temp)
            image_temp = image_temp.astype(np.float32)

            test_image = image_temp[:, :, 0] / 255.0
            test_image = np.expand_dims(test_image, axis=(0, 1))
            with torch.no_grad():
                prediction = model(torch.from_numpy(test_image.astype(np.float32)).to(DEVICE))

            prediction = prediction.cpu().numpy()
            prediction = np.reshape(prediction, sum(num for num in word_count))
            prediction = np.argmax(prediction)

            get_cls = test_cls[i]

            if int(get_cls) == int(prediction):
                count += 1

        print(f'Accuracy: {count / (sum(num for num in word_count)) * 100:.2f}\n')
'''
CNN
iter_num = 100000; batch_size = 64
Accuracy : 91.45
  Loss   :  0.00784
'''