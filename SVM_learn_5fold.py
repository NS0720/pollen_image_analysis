import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv(r"D:\python_working\pollen_statistics\radius_features_with_center_pattern_004_1.csv")

target_classes = ["cedar", "cypre", "dust"]
df = df[df["type"].isin(target_classes)]

# ラベルを数値に変換（scikit-learn用）
label_map = {name: i for i, name in enumerate(target_classes)}
df['label'] = df['type'].map(label_map)

# 特徴量とラベル
X = df[['radius','circularity','spike_blur_3*3','offset_ratio_comb']]
y = df['label']

# 標準化（CV内でやるのが本来ですが、パイプラインで簡単に書く方法も）
from sklearn.pipeline import make_pipeline
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

# 5-fold Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# クロスバリデーションで全データの予測値を得る（各foldでテストデータになったときの予測）
y_pred = cross_val_predict(model, X, y, cv=cv)

# レポート・混同行列
print("📋 Classification Report (5-fold CV):\n")
print(classification_report(y, y_pred, target_names=label_map.keys()))

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title("Confusion Matrix (SVM, 5-fold CV, 4 features)")
plt.tight_layout()

plt.savefig("confusion_matrix_svm_5foldCV_4features.png", dpi=300)
plt.savefig("confusion_matrix_svm_5foldCV_4features.pdf", bbox_inches='tight')
plt.show()
