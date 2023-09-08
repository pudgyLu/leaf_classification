# leaf_classification
Kaggle: [Classify Leaves](https://www.kaggle.com/competitions/classify-leaves), Private score: 0.94181


## Data Aug

```python
# trans_mode (string): [
#           horizonflip, 
#           rotation, 
#           brightness, 
#           RandomVerticalFlip
#   ]
  transforms.RandomHorizontalFlip(p=1),  # 水平翻转
  transforms.RandomRotation(180, expand=True),  # 随机+-180旋转
  transforms.ColorJitter(brightness=0.5),  # 亮度
  transforms.RandomVerticalFlip(1),  # 垂直翻转
```

## Model
* pre-trained model
  * resnet34
  * resnet50
  * resnext50_32x4d

* finetune
  * fc initialize
  * fc initialize with xavier, 10x lr
  * freeze params without fc, 10x lr

* params

```python
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-3
num_epoch = 20
```

## Requirements
```shell
pip install -r requirements.txt
```


## Usage

```shell
python leaf_classify
```





