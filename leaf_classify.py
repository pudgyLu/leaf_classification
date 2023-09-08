import random
import os
from torch.utils.data import Dataset, DataLoader

from eda_leaf import *
from data_leaf import *
from predict import *
from model import *
from train import *

exp_name = 'exp7'
train_mode = 'fc10'  # 1) fc随机初始化，2）fc随机初始化，学习率是其他的10倍，3）fc随机初始化，学习率调大的10倍，冻结前面层

continue_train = True
model_path = './checkpoints/resnet_finetune_' + exp_name + '.ckpt'
saveFileName = './results/submission_' + exp_name + '.csv'

random.seed(2023)
torch.manual_seed(2023)

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
weight_decay = 1e-3
num_epoch = 20

data_path = '../../data/classify-leaves/train.csv'
labels_dataframe = pd.read_csv(data_path)

# EDA
# eda_leaf = EDALeaf(data_path)
# eda_leaf.basic_info()
# eda_leaf.img_show()

# label 处理
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))  # label转成对应的数字
num_to_class = {v: k for k, v in class_to_num.items()}  # 再转换回来，方便最后预测的时候使用

train_path = '../../data/classify-leaves/train.csv'
test_path = '../../data/classify-leaves/test.csv'
img_path = '../../data/classify-leaves/'  # csv文件中已经images的路径了，因此这里只到上一级目录

train_dataset = (
        LeavesData(class_to_num, train_path, img_path, data_mode='train', trans_mode='train')
        + LeavesData(class_to_num, train_path, img_path, data_mode='train', trans_mode='horizonflip')
        + LeavesData(class_to_num, train_path, img_path, data_mode='train', trans_mode='RandomVerticalFlip')
        + LeavesData(class_to_num, train_path, img_path, data_mode='train', trans_mode='brightness')
)
val_dataset = LeavesData(class_to_num, train_path, img_path, data_mode='valid', trans_mode='valid')
test_dataset = LeavesData(class_to_num, test_path, img_path, data_mode='test', trans_mode='test')


# dataloader
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5
)

val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5
)

test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5
)


model = res_model(n_classes, feature_extract=False)
# 继续训练
if continue_train:
    model.load_state_dict(torch.load(model_path))

model = model.to(device)
model.device = device

# lossfunc: cross-entropy
criterion = nn.CrossEntropyLoss()

# Initialize optimizer
if train_mode == 'fc':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, weight_decay=weight_decay
    )

elif train_mode == 'fc10':
    # 除了最后一层的learning rate外，用的是默认的learning rate
    # 最后一层的learning rate用的是十倍的learning rate
    params_lx = [
        param for name, param in model.named_parameters()
        if name not in ["fc.weight", "fc.bias"]]  # 非最后一层的参数
    optimizer = torch.optim.SGD([
        {'params': params_lx},
        {'params': model.fc.parameters(), 'lr': learning_rate * 10}],
        lr=learning_rate, weight_decay=0.001)
else:
    trainer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)


# The number of training epochs.
n_epochs = num_epoch

# train
train_ft(model, train_loader, val_loader, optimizer, criterion, n_epochs, model_path, device)

# inference
predict_leaf = PredictLeaf(test_path, test_loader, num_to_class, saveFileName, model_path)
predict_leaf.eval()
