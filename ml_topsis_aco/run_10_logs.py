


"""
Try a different classifier, such as Random Forest, which might handle the multi-label classification better:
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf.fit(X_train_scaled, y_train)
predictions = rf.predict(X_test_scaled)

"""


"""
Logs

Loading dataset...
bibtex:train - exists, not redownloading
bibtex:test - exists, not redownloading
Preprocessing data...
Starting feature selection...
Selecting top 550 features
Feature statistics:
Mean of X_train: 0.03733175213486346
Std of X_train: 0.18957344860872402
Min of X_train: 0.0
Max of X_train: 1.0
Number of non-zero elements in X_train: 456141
Sparsity of X_train: 0.9626682478651365
Mean of y_train: 0.015052757419824316
Std of y_train: 0.12176276899727699
Min of y_train: 0
Max of y_train: 1
Performing MTOPSIS feature ranking...
Input shape: X=(6655, 1836), y=(6655, 159)
Feature-label correlation shape: (1836,)
Feature-label correlation range: [0.00252147898364939, 0.025933492569921068]
Feature-feature correlation shape: (1836, 1836)
Weights shape: (1836,)
Weights range: [0.00031578732185150665, 0.000959495107004753]
Weighted decision matrix shape: (1836,)
Weighted decision matrix range: [2.159880501667457e-06, 1.2597636946018777e-05]
Separation measures shape: best (1836,), worst (1836,)
Relative closeness shape: (1836,)
Relative closeness range: [0.0, 0.9999904194888545]
Number of features selected by MTOPSIS: 550
Top 5 feature indices: [1117 1823  951 1129 1428]
Relative closeness values for top 5 features: [0.99999042 0.94529446 0.94529446 0.89930608 0.87950366]
Performing Modified ACO feature reranking...
Number of selected features: 550
ACO Iterations: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [09:38<00:00, 115.62s/it]
Number of features selected by ACO: 238
Top 5 feature indices: [1175 1261 1751 1378   50]
Feature selection complete. Training model...
Making predictions...
Evaluating model...
Accuracy of the model: 9.05%
Hamming Loss: 0.0142
Average Precision score: 0.1718
Number of selected features: 238
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                    Accuracy ▁
wandb:           Average Precision ▁
wandb:                Hamming Loss ▁
wandb: Number of Selected Features ▁
wandb: 
wandb: Run summary:
wandb:                    Accuracy 0.09054
wandb:           Average Precision 0.17177
wandb:                Hamming Loss 0.01417
wandb: Number of Selected Features 238
wandb: 
wandb: 🚀 View run bibtex-ml-classification at: https://wandb.ai/sharhanalhassan-university-of-ghana/ML-TOPSIS-ACO-SMALL-RVM/runs/sczr8di0
wandb: ⭐️ View project at: https://wandb.ai/sharhanalhassan-university-of-ghana/ML-TOPSIS-ACO-SMALL-RVM
wandb: Synced 5 W&B file(s), 1 media file(s), 2 artifact file(s) and 0 other file(s)
"""