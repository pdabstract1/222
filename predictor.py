# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt

# 从 LIME 库中导入 LimeTabularExplainer，用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer


# =========================
#       加载模型与数据
# =========================

model = joblib.load('RF.pkl')
X_test = pd.read_csv('X_test.csv')

feature_names = [
    "RR",  # 呼吸频率
    "YS",  # 黄染
    "Fever",  # 发热
    "PCT",  # 降钙素原
    "NC",  # 鼻塞
    "AFT",  # 流产
    "WBC",  # 白细胞
]


st.title("新生儿早发型败血症预测器")


# =========================
#       输入表单
# =========================
with st.form("prediction_form"):
    st.subheader("请输入患者信息")

    RR = st.number_input("呼吸频率:", min_value=0, max_value=120, value=62)
    YS = st.selectbox("黄染:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    Fever = st.selectbox("发热:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    PCT = st.number_input("降钙素原:", min_value=0.00, max_value=100.00, value=1.75)
    NC = st.selectbox("鼻塞:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    AFT = st.selectbox("流产:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")
    WBC = st.number_input("白细胞:", min_value=0.00, max_value=120.00, value=25.27)

    submitted = st.form_submit_button("Predict")


# =========================
#       执行预测
# =========================
if submitted:
    feature_values = [RR, YS, Fever, PCT, NC, AFT, WBC]
    features = np.array([feature_values])

    # 预测类别与概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.subheader("预测结果")
    class_label = "患病 (1)" if predicted_class == 1 else "未患病 (0)"
    st.write(f"**预测类别:** {class_label}")
    st.write(f"**预测概率:** {predicted_proba}")

    # 生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        st.write(
            f"模型预测患病概率为 {probability:.1f}%。建议进一步进行医学评估与干预。"
        )
    else:
        st.write(
            f"模型预测未患病概率为 {probability:.1f}%。仍建议保持健康并定期检查。"
        )


    # =========================
    #       SHAP 力图
    # =========================
    st.subheader("SHAP 力解释图")

    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(
        pd.DataFrame([feature_values], columns=feature_names)
    )

    plt.figure(figsize=(10, 6))

    if predicted_class == 1:
        shap.force_plot(
            explainer_shap.expected_value[1],
            shap_values[:, :, 1],
            pd.DataFrame([feature_values], columns=feature_names),
            matplotlib=True, show=False
        )
    else:
        shap.force_plot(
            explainer_shap.expected_value[0],
            shap_values[:, :, 0],
            pd.DataFrame([feature_values], columns=feature_names),
            matplotlib=True, show=False
        )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP 力解释图')


    # =========================
    #       LIME 解释
    # =========================
    st.subheader("LIME 解释")

    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['未患病', '患病'],
        mode='classification'
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)

