import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class EDALeaf:
    def __init__(self, data_path):
        self.labels_dataframe = pd.read_csv(data_path)

    def basic_info(self):
        self.labels_dataframe.head(5)
        self.labels_dataframe.describe()

    def barw(self, ax):
        for p in ax.patches:
            val = p.get_width()  # height of the bar
            x = p.get_x() + p.get_width()  # x- position
            y = p.get_y() + p.get_height() / 2  # y-position
            ax.annotate(round(val, 2), (x, y))

    def img_show(self):
        # finding top leaves
        plt.figure(figsize=(15, 30))
        ax0 = sns.countplot(y=self.labels_dataframe['label'], order=self.labels_dataframe['label'].value_counts().index)
        self.barw(ax0)
        plt.show()



















