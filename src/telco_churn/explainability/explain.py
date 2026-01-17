import numpy as np
import shap

def build_explainer(*, pipe):
    clf = pipe.named_steps["clf"]
    return shap.TreeExplainer(clf)

def top_feature_name(*, pipe, feature_names, explainer, X_row):
    X_t = pipe.named_steps["pre"].transform(pipe.named_steps["spec"].transform(X_row))
    sv_row = np.asarray(explainer.shap_values(X_t))[0]
    return feature_names[int(np.argmax(np.abs(sv_row)))]
