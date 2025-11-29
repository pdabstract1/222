# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# 初始化 session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'predicted_proba' not in st.session_state:
    st.session_state.predicted_proba = None
if 'advice' not in st.session_state:
    st.session_state.advice = None
if 'shap_plot_generated' not in st.session_state:
    st.session_state.shap_plot_generated = False

# 加载模型和数据
model = joblib.load('RF.pkl')
X_test = pd.read_csv('X_test.csv')

feature_names = [
    "RR", "YS", "Fever", "PCT", "NC", "AFT", "WBC",
]

st.title("新生儿早发型败血症预测器")

# 🔴 修改：不使用表单，直接使用输入组件
st.subheader("请输入患者信息")

# 所有输入组件都在表单外
RR = st.number_input("呼吸频率:", min_value=0, max_value=120, value=62)
YS = st.selectbox("黄染:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
Fever = st.selectbox("发热:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
PCT = st.number_input("降钙素原:", min_value=0.00, max_value=100.00, value=1.75)
NC = st.selectbox("鼻塞:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
AFT = st.selectbox("流产:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
WBC = st.number_input("白细胞:", min_value=0.00, max_value=120.00, value=25.27)

# 预测按钮
if st.button("Predict"):
    # 处理输入数据并进行预测
    feature_values = [RR, YS, Fever, PCT, NC, AFT, WBC]
    features = np.array([feature_values])

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.predicted_proba = predicted_proba
    st.session_state.feature_values = feature_values
    st.session_state.features = features

    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您患有心脏病的风险较高。 "
            f"模型预测您患有心脏病的概率为 {probability:.1f}%。 "
            "建议您咨询医疗保健提供者进行进一步评估和可能的干预。"
        )
    else:
        advice = (
            f"根据我们的模型，您患有心脏病的风险较低。 "
            f"模型预测您未患有心脏病的概率为 {probability:.1f}%。 "
            "然而，保持健康的生活方式很重要。请继续定期与您的医疗保健提供者进行体检。"
        )

    st.session_state.advice = advice
    st.session_state.shap_plot_generated = False

    st.success("预测完成！")

# 显示预测结果（其余代码保持不变）
if st.session_state.prediction_made:
    st.subheader("预测结果")
    class_label = "患病 (1)" if st.session_state.predicted_class == 1 else "未患病 (0)"
    st.write(f"**预测类别:** {class_label}")
    st.write(f"**预测概率:** {st.session_state.predicted_proba}")
    st.write(st.session_state.advice)

    # SHAP 和 LIME 解释代码保持不变...
