import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overview(self):
        print(f"Shape: {self.df.shape}")
        print("\nDtypes:\n", self.df.dtypes)

    def missing_summary(self):
        miss = self.df.isnull().sum()
        print(miss[miss > 0])

    def target_summary(self, target):
        print(self.df[target].value_counts())
        print(f"Survival Rate: {self.df[target].mean():.2f}")

    def corr_heatmap(self):
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.show()