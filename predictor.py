


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

# 🔴 新增开始：初始化 session state
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
# 🟢 新增结束

# 加载训练好的随机森林模型（RF.pkl）
model = joblib.load('RF.pkl')

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')

# # 定义特征名称，对应数据集中的列名
# feature_names = [
#     "age",  # 年龄
#     "sex",  # 性别
#     "cp",  # 胸痛类型
#     "trestbps",  # 静息血压
#     "chol",  # 血清胆固醇
#     "fbs",  # 空腹血糖
#     "restecg",  # 静息心电图结果
#     "thalach",  # 最大心率
#     "exang",  # 运动诱发心绞痛
#     "oldpeak",  # 运动相对于静息的 ST 段抑制
#     "slope",  # ST 段的坡度
#     "ca",  # 主要血管数量（通过荧光造影测量）
#     "thal"  # 地中海贫血（thalassemia）类型
# ]

# 定义特征名称，对应数据集中的列名
feature_names = [
    "RR",  # 呼吸频率
    "PCT",  # 降钙素原
    "WBC",  # 白细胞
    "YS",  # 黄染
    "Fever",  # 发热
    "NC",  # 鼻塞
    "AFT",  # 流产
]

# Streamlit 用户界面
st.title("新生儿早发型败血症预测器")  # 设置网页标题

# 🔴 新增开始：使用表单来组织输入，防止重新运行
with st.form("prediction_form"):
    st.subheader("请输入患者信息")
    # 🟢 新增结束

    # 呼吸频率：数值输入框
    RR = st.number_input("呼吸频率:", min_value=0, max_value=120, value=41)

    # 降钙素原：数值输入框
    PCT = st.number_input("降钙素原:", min_value=0.000, max_value=100.000, value=1.001)

    # 白细胞：数值输入框
    WBC = st.number_input("白细胞:", min_value=0.00, max_value=120.00, value=6.00)

    # 黄染：分类选择框（0：否，1：是）
    YS = st.selectbox("黄染:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

    # 发热：分类选择框（0：否，1：是）
    Fever = st.selectbox("发热:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

    # 性别：分类选择框（0：否，1：是）
    NC = st.selectbox("鼻塞:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

    # 性别：分类选择框（0：否，1：是）
    AFT = st.selectbox("流产:", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")

    # 🔴 新增开始：提交按钮
    submitted = st.form_submit_button("Predict")
# 🟢 新增结束

# 🔴 修改开始：当用户点击 "Predict" 按钮时执行以下代码（修改了条件判断）
if submitted:
    # 处理输入数据并进行预测
    feature_values = [RR, PCT, WBC, YS, Fever, NC, AFT]  # 将用户输入的特征值存入列表
    features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

    # 预测类别（0：无败血症，1：有败血症）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 🔴 新增开始：保存预测结果到 session state
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.predicted_proba = predicted_proba
    st.session_state.feature_values = feature_values
    st.session_state.features = features

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您患有心脏病的风险较高。 "
            f"模型预测您患有心脏病的概率为 {probability:.1f}%。 "
            "建议您咨询医疗保健提供者进行进一步评估和可能的干预。"
        )
    # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"根据我们的模型，您患有心脏病的风险较低。 "
            f"模型预测您未患有心脏病的概率为 {probability:.1f}%。 "
            "然而，保持健康的生活方式很重要。请继续定期与您的医疗保健提供者进行体检。"
        )

    st.session_state.advice = advice
    st.session_state.shap_plot_generated = False

    # 显示成功消息
    st.success("预测完成！")
# 🟢 新增结束

# 🔴 新增开始：显示预测结果（如果存在）
if st.session_state.prediction_made:
    st.subheader("预测结果")

    # 显示预测结果
    class_label = "患病 (1)" if st.session_state.predicted_class == 1 else "未患病 (0)"
    st.write(f"**预测类别:** {class_label}")
    st.write(f"**预测概率:** {st.session_state.predicted_proba}")

    # 显示建议
    st.write(st.session_state.advice)

    # SHAP 解释
    st.subheader("SHAP 力解释图")

    # 只在第一次或需要重新生成时创建 SHAP 图
    if not st.session_state.shap_plot_generated:
        # 创建 SHAP 解释器，基于树模型（如随机森林）
        explainer_shap = shap.TreeExplainer(model)
        # 计算 SHAP 值，用于解释模型的预测
        shap_values = explainer_shap.shap_values(pd.DataFrame([st.session_state.feature_values], columns=feature_names))

        # 根据预测类别显示 SHAP 强制图
        plt.figure(figsize=(10, 6))
        if st.session_state.predicted_class == 1:
            shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1],
                            pd.DataFrame([st.session_state.feature_values], columns=feature_names),
                            matplotlib=True, show=False)
        else:
            shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0],
                            pd.DataFrame([st.session_state.feature_values], columns=feature_names),
                            matplotlib=True, show=False)

        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.session_state.shap_plot_generated = True

    # 显示已保存的 SHAP 图
    st.image("shap_force_plot.png", caption='SHAP 力解释图')

    # LIME 解释
    st.subheader("LIME 解释")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['未患病', '患病'],  # 调整类别名称以匹配分类任务
        mode='classification'
    )

    # 解释实例
    lime_exp = lime_explainer.explain_instance(
        data_row=st.session_state.features.flatten(),
        predict_fn=model.predict_proba
    )

    # 显示 LIME 解释，不包含特征值表格
    lime_html = lime_exp.as_html(show_table=False)  # 禁用特征值表格
    st.components.v1.html(lime_html, height=800, scrolling=True)

    # 🔴 新增开始：添加清除结果的按钮
    if st.button("清除预测结果"):
        st.session_state.prediction_made = False
        st.session_state.predicted_class = None
        st.session_state.predicted_proba = None
        st.session_state.advice = None
        st.session_state.shap_plot_generated = False
        st.rerun()
# 🟢 新增结束

