This repository contains a comprehensive pipeline for predicting chemical properties—including toxicity, flammability, reactivity, and reactivity with water—based on structured and graph-based representations of molecular data.
Dataset

🧠 Modeling
We implemented a diverse set of machine learning algorithms, including:

Traditional models: Random Forest (RF), k-Nearest Neighbors (KNN), Decision Tree (DT), XGBoost, and Support Vector Classifier (SVC)

Graph-based models: Graph Attention Networks (GAT), GraphSAGE, and Graph Convolutional Networks (GCN)

All models were optimized using a grid search strategy for hyperparameter tuning.

📊 Model Interpretability
To interpret model predictions, we applied:

SHAP (SHapley Additive exPlanations)

ICE plots (Individual Conditional Expectation)

Partial Dependence Plots (2D and 3D PDP)

⚗️ Model Evaluation and Validation
The application domain and Y-scrambling analysis were implemented in interactive Jupyter Notebooks to ensure robustness and reliability of the models.

🌐 Web Application
We also developed a user-friendly web interface for model inference and visualization.
