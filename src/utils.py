import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()

def plot_target_distribution(df, target_col):
    sns.countplot(x=target_col, data=df)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig("target_distribution.png")
    plt.close()
