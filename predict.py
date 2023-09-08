import torch
import pandas as pd

from tqdm import tqdm
from model import res_model


class PredictLeaf(object):

    def __init__(self, test_path, test_loader, num_to_class,
                 saveFileName='./submission.csv', model_path='./pre_res_model.ckpt'):
        self.saveFileName = saveFileName
        self.num_to_class = num_to_class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = res_model(176)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.test_loader = test_loader
        self.test_path = test_path

    def eval(self):
        # Make sure the model is in eval mode.
        # Some modules like Dropout or BatchNorm affect if the model is in training mode.
        self.model.eval()

        # Initialize a list to store the predictions.
        predictions = []
        # Iterate the testing set by batches.
        for batch in tqdm(self.test_loader):
            imgs = batch
            with torch.no_grad():
                logits = self.model(imgs.to(self.device))
            # Take the class with greatest logit as prediction and record it.
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

        preds = []
        for i in predictions:
            preds.append(self.num_to_class[i])

        test_data = pd.read_csv(self.test_path)
        test_data['label'] = pd.Series(preds)
        submission = pd.concat([test_data['image'], test_data['label']], axis=1)
        submission.to_csv(self.saveFileName, index=False)
        print("Done!")


if __name__ == '__main__':
    test_path = ''
    test_loader = ''
    num_to_class = ''
    model_path = ''
    predict_leaf = PredictLeaf(test_path, test_loader, num_to_class, model_path)
    predict_leaf.eval()

