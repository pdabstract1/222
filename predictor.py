import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# -------------------- Session State Initialization --------------------
def init_session_state():
    defaults = {
        'prediction_made': False,
        'predicted_class': None,
        'predicted_proba': None,
        'advice': None,
        'shap_plot_generated': False,
        'feature_values': None,
        'features': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# -------------------- Load Model and Test Data --------------------
model = joblib.load('RF.pkl')
X_test = pd.read_csv('X_test.csv')

feature_names = ["RR", "YS", "Fever", "PCT", "NC", "AFT", "WBC"]

# -------------------- UI --------------------
st.title("新生儿早发型败血症预测器（优化版）")

# Use form to prevent rerun problems
with st.form("prediction_form"):
    st.subheader("请输入患者信息")

    RR = st.number_input("呼吸频率", min_value=0, max_value=120, value=62, key="RR")
    YS = st.selectbox("黄染", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="YS")
    Fever = st.selectbox("发热", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="Fever")
    PCT = st.number_input("降钙素原", min_value=0.0, max_value=100.0, value=1.75, key="PCT")
    NC = st.selectbox("鼻塞", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="NC")
    AFT = st.selectbox("流产", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="AFT")
    WBC = st.number_input("白细胞", min_value=0.0, max_value=200.0, value=25.27, key="WBC")

    submitted = st.form_submit_button("Predict")

# -------------------- Prediction Execution --------------------
if submitted:
    feature_values = [RR, YS, Fever, PCT, NC, AFT, WBC]
    features = np.array([feature_values])

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Save session state
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.predicted_proba = predicted_proba
    st.session_state.feature_values = feature_values
    st.session_state.features = features
    st.session_state.shap_plot_generated = False

    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"模型预测风险偏高，患病概率为 {probability:.1f}%。"
            " 建议尽快咨询专业医生进行进一步检查。"
        )
    else:
        advice = (
            f"模型预测风险较低，未患病概率为 {probability:.1f}%。"
            " 但仍需保持观察，必要时可做进一步检查。"
        )

    st.session_state.advice = advice
    st.success("预测完成！")

# -------------------- Display Results --------------------
if st.session_state.prediction_made:

    st.subheader("预测结果")

    class_label = "患病 (1)" if st.session_state.predicted_class == 1 else "未患病 (0)"
    st.write(f"**预测类别:** {class_label}")
    st.write(f"**预测概率:** {st.session_state.predicted_proba}")
    st.write(st.session_state.advice)

    # -------------------- SHAP --------------------
    st.subheader("SHAP 力解释图")

    if not st.session_state.shap_plot_generated:
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(
            pd.DataFrame([st.session_state.feature_values], columns=feature_names)
        )

        plt.figure(figsize=(10, 6))

        cls = st.session_state.predicted_class
        shap.force_plot(
            explainer_shap.expected_value[cls],
            shap_values[cls][0],
            pd.DataFrame([st.session_state.feature_values], columns=feature_names),
            matplotlib=True,
            show=False
        )

        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.session_state.shap_plot_generated = True

    st.image("shap_force_plot.png", caption="SHAP 力解释图")

    # -------------------- LIME --------------------
    st.subheader("LIME 解释")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['未患病', '患病'],
        mode='classification'
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=st.session_state.features.flatten(),
        predict_fn=model.predict_proba
    )

    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)
