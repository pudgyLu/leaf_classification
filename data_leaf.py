import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LeavesData(Dataset):
    def __init__(self, class_to_num, csv_path, file_path,
                 data_mode='train', trans_mode='train',
                 valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            data_mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
            data_mode (string): [train, valid, test]
            trans_mode (string): [horizonflip, rotation, brightness, RandomVerticalFlip]
        """

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.data_mode = data_mode
        self.trans_mode = trans_mode

        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))
        self.class_to_num = class_to_num

        if data_mode == 'train':
            # 第一列包含图像文件的名称
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 0])  # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            # 第二列是图像的 label
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif data_mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif data_mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(data_mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置转换变量，还可以包括一系列的nomarlize等等操作
        if self.trans_mode == 'horizonflip':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1),  # 水平翻转
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 与预训练模型保持一致

            ])
        elif self.trans_mode == 'rotation':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(180, expand=True),  # 随机+-180旋转
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 与预训练模型保持一致
            ])
        elif self.trans_mode == 'brightness':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.5),  # 亮度
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 与预训练模型保持一致
            ])
        elif self.trans_mode == 'RandomVerticalFlip':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(1),  # 垂直翻转
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 与预训练模型保持一致
            ])

        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 与预训练模型保持一致
            ])

        img_as_img = transform(img_as_img)

        if self.data_mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = self.class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len
