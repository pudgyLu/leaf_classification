import torch
import torch.nn as nn
import torchvision.models as models


# 是否要冻住模型的前面一些层/冻住全部层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        # for param in model.parameters():
        #     param.requires_grad = False
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False


# resnet34/resnet50/resnext50_32x4d pre-train 模型
def res_model(num_classes, feature_extract=False, use_pretrained=True, continue_train=False):
    # model_ft = models.resnet34(pretrained=use_pretrained)
    # model_ft = models.resnet50(pretrained=use_pretrained)
    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    # 修改预训练模型的最后一层结构，输出修改为当前树叶分类数
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # nn.Sequential(nn.Linear(num_ftrs, num_classes))
    nn.init.xavier_uniform_(model_ft.fc.weight)  # 只对最后一层的weight做xavier初始化

    return model_ft
