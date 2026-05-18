import shap
import numpy as np

def shapValue(model, XBackground, XTest, bandList, typeChart):
    explainer = shap.GradientExplainer(model, XBackground)
    # (sample, timeStamps, features, outputs shape) -> (features)
    shapValues = np.mean(explainer(XTest).values.squeeze(-1),axis=1)
    barImage = shap.Explanation(
        values= shapValues.squeeze(),
        feature_names= bandList
    )

    shap.summary_plot(shapValues,feature_names= bandList, plot_type="bar")
    return barImage

