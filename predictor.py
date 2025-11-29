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

# 加载训练好的随机森林模型（RF.pkl）
model = joblib.load('RF.pkl')  

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')  

# 定义特征名称，对应数据集中的列名
feature_names = [  
    "age",       # 年龄  
    "sex",       # 性别  
    "cp",        # 胸痛类型  
    "trestbps",  # 静息血压  
    "chol",      # 血清胆固醇  
    "fbs",       # 空腹血糖  
    "restecg",   # 静息心电图结果  
    "thalach",   # 最大心率  
    "exang",     # 运动诱发心绞痛  
    "oldpeak",   # 运动相对于静息的 ST 段抑制  
    "slope",     # ST 段的坡度  
    "ca",        # 主要血管数量（通过荧光造影测量）  
    "thal"       # 地中海贫血（thalassemia）类型  
]  

# Streamlit 用户界面
st.title("心脏病预测器")  # 设置网页标题

# 年龄：数值输入框
age = st.number_input("年龄:", min_value=0, max_value=120, value=41)  

# 性别：分类选择框（0：女性，1：男性）
sex = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")  

# 胸痛类型（cp）：分类选择框（0-3）
cp = st.selectbox("胸痛类型 (CP):", options=[0, 1, 2, 3])  

# 静息血压（trestbps）：数值输入框
trestbps = st.number_input("静息血压 (trestbps):", min_value=50, max_value=200, value=120)  

# 血清胆固醇（chol）：数值输入框
chol = st.number_input("胆固醇 (chol):", min_value=100, max_value=600, value=157)  

# 空腹血糖 > 120 mg/dl（fbs）：分类选择框（0：否，1：是）
fbs = st.selectbox("空腹血糖 > 120 mg/dl (FBS):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")  

# 静息心电图结果（restecg）：分类选择框（0-2）
restecg = st.selectbox("静息心电图 (restecg):", options=[0, 1, 2])  

# 最大心率（thalach）：数值输入框
thalach = st.number_input("最大心率 (thalach):", min_value=60, max_value=220, value=182)  

# 运动诱发心绞痛（exang）：分类选择框（0：否，1：是）
exang = st.selectbox("运动诱发心绞痛 (exang):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否")  

# 运动引起的 ST 段抑制（oldpeak）：数值输入框
oldpeak = st.number_input("运动引起的 ST 段抑制 (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)  

# 运动峰值 ST 段的坡度（slope）：分类选择框（0-2）
slope = st.selectbox("运动峰值 ST 段的坡度 (slope):", options=[0, 1, 2])  

# 主要血管数量（通过荧光造影测量）（ca）：分类选择框（0-4）
ca = st.selectbox("主要血管数量（荧光造影测量）(ca):", options=[0, 1, 2, 3, 4])  

# 地中海贫血（thal）：分类选择框（0-3）
thal = st.selectbox("地中海贫血 (thal):", options=[0, 1, 2, 3])  

# 处理输入数据并进行预测
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
     # 预测类别（0：无心脏病，1：有心脏病）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

     # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)

    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'],  # Adjust class names to match your classification task
        mode='classification'
    )
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)