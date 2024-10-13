# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, accuracy_score
# from skopt import Optimizer
# from skopt.space import Integer
# from tqdm import tqdm
# import plotly.graph_objects as go
# import numpy as np
#
# # Read the dataset
# df = pd.read_excel("Original.xlsx")
#
# # Features and labels
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
#
# # Encode labels if they're categorical
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)
#
# # Split the dataset
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
#
# # Scale the data
# scaler = StandardScaler()
# X_train_s = scaler.fit_transform(X_train)
# X_val_s = scaler.transform(X_val)
# X_test_s = scaler.transform(X_test)
#
# # Define search space for Bayesian optimization
# space = [
#     Integer(150, 300, name='n_estimators'),
#     Integer(5, 30, name='max_depth'),
#     Integer(2, 20, name='min_samples_split'),
#     Integer(1, 10, name='min_samples_leaf'),
#     Integer(1, 16, name='max_features')
# ]
#
#
# # Define evaluation function for optimization
# def rf_evaluate(params):
#     rf = RandomForestClassifier(
#         n_estimators=params[0],
#         max_depth=params[1],
#         min_samples_split=params[2],
#         min_samples_leaf=params[3],
#         max_features=params[4],
#         random_state=0
#     )
#     rf.fit(X_train_s, y_train)
#     val_acc = accuracy_score(y_val, rf.predict(X_val_s))
#     return -val_acc
#
#
# # Initialize Bayesian optimizer
# opt = Optimizer(dimensions=space, random_state=42)
# n_calls = 20
#
# # Bayesian optimization process
# pbar = tqdm(total=n_calls, desc="Optimization Progress", unit="call")
# for i in range(n_calls):
#     next_point = opt.ask()
#     f_val = rf_evaluate(next_point)
#     opt.tell(next_point, f_val)
#     pbar.update(1)
# pbar.close()
#
# # Best parameters
# best_params = opt.get_result().x
# print(f'Best Parameters: {best_params}')
#
# # Train the best model
# rf_best = RandomForestClassifier(
#     n_estimators=best_params[0],
#     max_depth=best_params[1],
#     min_samples_split=best_params[2],
#     min_samples_leaf=best_params[3],
#     max_features=best_params[4],
#     random_state=0
# )
# rf_best.fit(X_train_s, y_train)
#
# # Prepare to store results
# results = []
#
# # Error Analysis Scenarios
# # 1. Data Missing Imputation using KNN
# imputer = KNNImputer()
# X_train_imputed = imputer.fit_transform(X_train_s)
# X_test_imputed = imputer.transform(X_test_s)
#
# rf_best.fit(X_train_imputed, y_train)
# y_pred = rf_best.predict(X_test_imputed)
#
# cm = confusion_matrix(y_test, y_pred)
# if cm.shape == (2, 2):
#     tn, fp, fn, tp = cm.ravel()
#     overestimation = fp  # False Positive
#     underestimation = fn  # False Negative
#
#     results.append({
#         'Error_Type': 'Data Missing',
#         'Method': 'KNN Imputation',
#         'Overestimation': overestimation,
#         'Underestimation': underestimation
#     })
#
# # 2. Feature Limitation
# num_features = X_train_s.shape[1]
# feature_reduction_ratios = [0.01, 0.1, 0.3, 0.5]
#
# for ratio in feature_reduction_ratios:
#     num_features_to_keep = int(num_features * (1 - ratio))
#     X_train_limited = X_train_s[:, :num_features_to_keep]
#     X_test_limited = X_test_s[:, :num_features_to_keep]
#
#     rf_best.fit(X_train_limited, y_train)
#     y_pred = rf_best.predict(X_test_limited)
#
#     cm = confusion_matrix(y_test, y_pred)
#     if cm.shape == (2, 2):
#         tn, fp, fn, tp = cm.ravel()
#
#         overestimation = fp
#         underestimation = fn
#
#         results.append({
#             'Error_Type': 'Feature Limitation',
#             'Feature_Reduction_Ratio': f'{int(ratio * 100)}%',
#             'Overestimation': overestimation,
#             'Underestimation': underestimation
#         })
#
#     # 3. Dataset Size Variation
# dataset_size_ratios = [0.2, 0.4, 0.6, 0.8, 0.9]
#
# for size_ratio in dataset_size_ratios:
#     X_train_small, _, y_train_small, _ = train_test_split(X_train_s, y_train, train_size=size_ratio, random_state=42)
#
#     rf_best.fit(X_train_small, y_train_small)
#     y_pred = rf_best.predict(X_test_s)
#
#     cm = confusion_matrix(y_test, y_pred)
#     if cm.shape == (2, 2):
#         tn, fp, fn, tp = cm.ravel()
#
#         overestimation = fp
#         underestimation = fn
#
#         results.append({
#             'Error_Type': 'Dataset Size',
#             'Dataset_Size_Ratio': f'{int(size_ratio * 100)}%',
#             'Overestimation': overestimation,
#             'Underestimation': underestimation
#         })
#
#     # 4. Data Heterogeneity (Preprocessing) using only StandardScaler
# # Since StandardScaler is already applied, we will just document this step
# rf_best.fit(X_train_s, y_train)
# y_pred = rf_best.predict(X_test_s)
#
# cm = confusion_matrix(y_test, y_pred)
# if cm.shape == (2, 2):
#     tn, fp, fn, tp = cm.ravel()
#
#     overestimation = fp
#     underestimation = fn
#
#     results.append({
#         'Error_Type': 'Data Heterogeneity',
#         'Method': 'Standardization',
#         'Overestimation': overestimation,
#         'Underestimation': underestimation
#     })
#
# # Save results to CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv('detailed_error_analysis_results.csv', index=False)
# print("Detailed error analysis results saved to detailed_error_analysis_results.csv")
#
# # Visualization (Sankey)
# nodes = ['Overestimation', 'Underestimation'] + [r['Error_Type'] for r in results if r['Error_Type']]
# source, target, values = [], [], []
#
# for error_type in results_df['Error_Type'].unique():
#     sub_df = results_df[results_df['Error_Type'] == error_type]
#     if not sub_df.empty:
#         overestimation_sum = sub_df['Overestimation'].sum()
#         underestimation_sum = sub_df['Underestimation'].sum()
#
#         if overestimation_sum > 0:
#             source.append(nodes.index(error_type))
#             target.append(nodes.index('Overestimation'))
#             values.append(overestimation_sum)
#
#         if underestimation_sum > 0:
#             source.append(nodes.index(error_type))
#             target.append(nodes.index('Underestimation'))
#             values.append(underestimation_sum)
#
# fig = go.Figure(go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=nodes
#     ),
#     link=dict(
#         source=source,
#         target=target,
#         value=values
#     )
# ))
#
# fig.update_layout(title_text="Error Types and Their Impact on Model Estimations", font_size=12)
# fig.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from skopt import Optimizer
from skopt.space import Integer
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np

# Read the dataset
df = pd.read_excel("try.xlsx")

# Features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels if they're categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Impute missing values
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

# Check that shapes are consistent
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Scale the data
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Define search space for Bayesian optimization
space = [
    Integer(150, 300, name='n_estimators'),
    Integer(5, 30, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf'),
    Integer(1, 16, name='max_features')
]

# Define evaluation function for optimization
def rf_evaluate(params):
    rf = RandomForestClassifier(
        n_estimators=params[0],
        max_depth=params[1],
        min_samples_split=params[2],
        min_samples_leaf=params[3],
        max_features=params[4],
        random_state=0
    )
    rf.fit(X_train_s, y_train)
    val_acc = accuracy_score(y_val, rf.predict(X_val_s))
    return -val_acc

# Initialize Bayesian optimizer
opt = Optimizer(dimensions=space, random_state=42)
n_calls = 50

# Bayesian optimization process
pbar = tqdm(total=n_calls, desc="Optimization Progress", unit="call")
for i in range(n_calls):
    next_point = opt.ask()
    f_val = rf_evaluate(next_point)
    opt.tell(next_point, f_val)
    pbar.update(1)
pbar.close()

# Best parameters
best_params = opt.get_result().x
print(f'Best Parameters: {best_params}')

# Train the best model
rf_best = RandomForestClassifier(
    n_estimators=best_params[0],
    max_depth=best_params[1],
    min_samples_split=best_params[2],
    min_samples_leaf=best_params[3],
    max_features=best_params[4],
    random_state=0
)
rf_best.fit(X_train_s, y_train)

# Prepare to store results
results = []

# 1. Data Missing Imputation using KNN
imputer = KNNImputer()
X_train_imputed = imputer.fit_transform(X_train_s)
X_test_imputed = imputer.transform(X_test_s)

rf_best.fit(X_train_imputed, y_train)
y_pred = rf_best.predict(X_test_imputed)

cm = confusion_matrix(y_test, y_pred)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    overestimation = fp  # False Positive
    underestimation = fn  # False Negative

    results.append({
        'Error_Type': 'Data Missing',
        'Method': 'KNN Imputation',
        'Overestimation': overestimation,
        'Underestimation': underestimation
    })

# 2. Feature Limitation
num_features = X_train_s.shape[1]
feature_reduction_ratios = [0.01, 0.1, 0.3, 0.5]

for ratio in feature_reduction_ratios:
    num_features_to_keep = int(num_features * (1 - ratio))
    X_train_limited = X_train_s[:, :num_features_to_keep]
    X_test_limited = X_test_s[:, :num_features_to_keep]

    rf_best.fit(X_train_limited, y_train)
    y_pred = rf_best.predict(X_test_limited)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        overestimation = fp
        underestimation = fn

        results.append({
            'Error_Type': 'Feature Limitation',
            'Feature_Reduction_Ratio': f'{int(ratio * 100)}%',
            'Overestimation': overestimation,
            'Underestimation': underestimation
        })

    # 3. Dataset Size Variation
dataset_size_ratios = [0.2, 0.4, 0.6, 0.8, 0.9]

for size_ratio in dataset_size_ratios:
    X_train_small, _, y_train_small, _ = train_test_split(X_train_s, y_train, train_size=size_ratio, random_state=42)

    rf_best.fit(X_train_small, y_train_small)
    y_pred = rf_best.predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        overestimation = fp
        underestimation = fn

        results.append({
            'Error_Type': 'Dataset Size',
            'Dataset_Size_Ratio': f'{int(size_ratio * 100)}%',
            'Overestimation': overestimation,
            'Underestimation': underestimation
        })

    # 4. Data Heterogeneity (Preprocessing) using only StandardScaler
# Since StandardScaler is already applied, we will just document this step
rf_best.fit(X_train_s, y_train)
y_pred = rf_best.predict(X_test_s)

cm = confusion_matrix(y_test, y_pred)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()

    overestimation = fp
    underestimation = fn

    results.append({
        'Error_Type': 'Data Heterogeneity',
        'Method': 'Standardization',
        'Overestimation': overestimation,
        'Underestimation': underestimation
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('detailed_error_analysis_results.csv', index=False)
print("Detailed error analysis results saved to detailed_error_analysis_results.csv")

# Visualization (Sankey)
nodes = ['Overestimation', 'Underestimation'] + [r['Error_Type'] for r in results if r['Error_Type']]
source, target, values = [], [], []

for error_type in results_df['Error_Type'].unique():
    sub_df = results_df[results_df['Error_Type'] == error_type]
    if not sub_df.empty:
        overestimation_sum = sub_df['Overestimation'].sum()
        underestimation_sum = sub_df['Underestimation'].sum()

        if overestimation_sum > 0:
            source.append(nodes.index(error_type))
            target.append(nodes.index('Overestimation'))
            values.append(overestimation_sum)

        if underestimation_sum > 0:
            source.append(nodes.index(error_type))
            target.append(nodes.index('Underestimation'))
            values.append(underestimation_sum)

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
))

fig.update_layout(title_text="Error Types and Their Impact on Model Estimations", font_size=12)
fig.show()