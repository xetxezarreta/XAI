# Explainable Artificial Intelligence (XAI) 

[Explainable AI (XAI)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence) refers to methods and techniques in the application of artificial intelligence technology (AI) such that the results of the solution can be understood by human experts. It contrasts with the concept of the "black box" in machine learning where even their designers cannot explain why the AI arrived at a specific decision.

2 explainability methods:
- **Model-specific (intrinsic)**: Model-specific interpretability methods are limited to specific model classes. Intrinsic methods are by definition model-specific.
- **Model-agnostic (post-hoc)**: Model-agnostic methods are not tied to a particular type of ML model. Model-agnostic interpretations are usually post-hoc. 

2 types of interpretation:
- **Local**: Explain how and why an specific prediction was made
- **Global**: Explain how a model works overall
  - What in average your model care about and how they impact the outcome
                                                  
| Libraries           | References             | Global/Local  | Model-specific/Model-agnostic   |
| ------------------- | ---------------------- | ------------- | ------------------------------- |
| **ELI5**            | ref1                   | G/L           | MS/MA                           |
| **LIME**            | ref2                   | L             | MA                              |
| **LIME For Time**   | ref3                   | L             | MA                              |
| **SHAP**            | ref4                   | G/L           | MS/MA                           |
| **Interpret**       | ref5                   | G/L           | MS/MA                           |

G = Global, L = Local, MS = Model-specific, MA = Model-agnostic

## Tools

### ELI5 (Explain Like I’m 5)
https://github.com/TeamHG-Memex/eli5  
ELI5 is a Python package which helps to debug machine learning classifiers and explain their predictions.
- Provides visualisations for **“white-box”** models (simple models: logistic regression, etc...) 
  - Local and global interpretation
- Provides **Permutation Importance** for **"black box"** models (random forest, boosting, etc...)
  - Only global interpretation

Supported libraries/frameworks: 
- Scikit-learn
- Keras
- Xgboost
- LightGBM
- CatBoost
- Lightning
- Sklearn-crfsuite

### LIME (Local Interpretable Model-Agnostic Explanations)
https://github.com/marcotcr/lime  
LIME is about explaining what machine learning classifiers (or models) are doing. LIME supports explaining individual predictions for text classifiers or classifiers that act on tables (numpy arrays of numerical or categorical data) or images, with a package called lime (short for local interpretable model-agnostic explanations). LIME is based on the work presented in [this paper](https://arxiv.org/abs/1602.04938).

Local Interpretable Model-Agnostic Explanations:
- Local = Explains why a single data point was classified as a specific class
- Model-Agnostic = Treats the model as a black-box. Doesn't need to know how it makes predictions

It doesn't matter what algorithms to use since it treats models like black boxes (Model-Agnostic)

### LIME For Time (SHapley Additive exPlanations)
https://github.com/emanuel-metzenthin/Lime-For-Time  
LIME-For-Time is a LIME algorithm applycation (LIME - Local Interpretable Model-Agnostic Explanations) developed by Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin to time series classification. The goal was to explain time series classification algorithms by highlighting the areas of the timeseries used by the classifier in its prediction. That way a user of the classification algorithm can determine if the algorithm can be trusted.

### SHAP (SHapley Additive exPlanations)
https://github.com/slundberg/shap  
SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only possible consistent and locally accurate additive feature attribution method based on expectations.

SHAP API
- Tree Explainer
  - For tree based models
  - Works with scikit-learn, xgboost, lightgbm, catboost
- Deep Explainer
  - For Deep Learning models
- Kernel Explainer
  - Model-agnostic explainer
  
### Interpret (InterpretML)
https://github.com/microsoft/interpret

## References
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8466590
