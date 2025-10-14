# ===== A1: Feature Correlation Analysis and Heatmap =====
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_correlation_heatmap(file_path):
    """Loads dataset and plots feature correlation heatmap."""
    df = pd.read_csv(file_path)
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

# ===== MAIN =====
if __name__ == "__main__":
    feature_correlation_heatmap("DCT_mal.csv")
