# -*- coding: utf-8 -*-
"""
重构版：Streamlit + SHAP 自动适配 + LIME
适用于 Jupyter Notebook (保存为 .py 用 streamlit run 运行)，
或直接放在 Streamlit app 中运行。
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import io
import os
import warnings

warnings.filterwarnings("ignore")

# -------------------- 辅助函数：初始化 session_state --------------------
def init_session_state():
    defaults = {
        "prediction_made": False,
        "predicted_class": None,
        "predicted_proba": None,
        "advice": None,
        "shap_plot_generated": False,
        "feature_values": None,
        "features": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# -------------------- 加载模型与测试数据 --------------------
# 修改为你自己的模型/路径
MODEL_PATH = "RF.pkl"
XTEST_PATH = "X_test.csv"

if not os.path.exists(MODEL_PATH):
    st.error(f"找不到模型文件: {MODEL_PATH}，请检查路径并上传模型。")
else:
    model = joblib.load(MODEL_PATH)

if not os.path.exists(XTEST_PATH):
    st.warning(f"找不到 {XTEST_PATH}，LIME 解释将不可用（或请提供一个训练集样本文件）。")
    X_test = None
else:
    X_test = pd.read_csv(XTEST_PATH)

# 特征名（请与模型训练时一致）
feature_names = ["RR", "YS", "Fever", "PCT", "NC", "AFT", "WBC"]

# -------------------- 页面标题 --------------------
st.title("新生儿早发型败血症预测器（重构版）")

# -------------------- 表单输入（使用 key 避免重置） --------------------
with st.form("prediction_form"):
    st.subheader("请输入患者信息")

    # 给每个控件加上 key，防止重跑时丢失
    RR = st.number_input("呼吸频率 (RR):", min_value=0, max_value=120, value=62, key="inp_RR")
    YS = st.selectbox("黄染 (YS):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="inp_YS")
    Fever = st.selectbox("发热 (Fever):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="inp_Fever")
    PCT = st.number_input("降钙素原 (PCT):", min_value=0.0, max_value=100.0, value=1.75, format="%.2f", key="inp_PCT")
    NC = st.selectbox("鼻塞 (NC):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="inp_NC")
    AFT = st.selectbox("流产 (AFT):", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", key="inp_AFT")
    WBC = st.number_input("白细胞 (WBC):", min_value=0.0, max_value=200.0, value=25.27, format="%.2f", key="inp_WBC")

    submitted = st.form_submit_button("Predict")

# -------------------- 预测与 session_state 更新 --------------------
if submitted:
    # 构造特征
    feature_values = [RR, YS, Fever, PCT, NC, AFT, WBC]
    features = np.array([feature_values])

    # 运行模型预测（在部署前请确保 model 已加载）
    try:
        predicted_class = int(model.predict(features)[0])
        predicted_proba = model.predict_proba(features)[0]
    except Exception as e:
        st.error(f"模型预测失败：{e}")
        predicted_class = None
        predicted_proba = None

    # 保存结果到 session_state（覆盖旧值）
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.predicted_proba = predicted_proba
    st.session_state.feature_values = feature_values
    st.session_state.features = features
    st.session_state.shap_plot_generated = False

    # 生成建议文本（示例）
    if predicted_proba is not None and predicted_class is not None:
        prob = predicted_proba[predicted_class] * 100
        if predicted_class == 1:
            advice = f"模型预测风险偏高，患病概率为 {prob:.1f}%。建议尽快就医检查。"
        else:
            advice = f"模型预测风险较低，未患病概率为 {prob:.1f}%。建议继续观察并定期复查。"
    else:
        advice = "无法生成建议（模型预测失败）。"

    st.session_state.advice = advice
    st.success("预测完成！")

# -------------------- SHAP：自动适配并绘图的辅助函数 --------------------
def generate_shap_plot_and_save(model, feature_values, feature_names, out_path="shap_force_plot.png"):
    """
    自动适配 shap_values 的不同格式并尝试绘制 force_plot（matplotlib=True）。
    如果 force_plot 不可用或出错，则回退到 waterfall/bar 图。
    返回 True/False 表示是否成功生成图片文件。
    """
    # 准备 DataFrame 单样本
    X_df = pd.DataFrame([feature_values], columns=feature_names)

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        # 如果是非树模型，可改用 KernelExplainer（非常慢）或直接跳过
        try:
            explainer = shap.Explainer(model, X_df)  # shap >=0.39 的通用接口
        except Exception as e:
            print("无法创建 SHAP explainer:", e)
            return False

    # 计算 shap_values（格式可能是 list 或 ndarray）
    try:
        shap_values = explainer.shap_values(X_df)
    except Exception as e:
        # shap 版本差异，尝试使用 explainer(X_df)
        try:
            ev = explainer(X_df)
            shap_values = ev.values
            expected_value = ev.base_values
        except Exception as ee:
            print("计算 SHAP 值失败:", e, ee)
            return False

    # 决定要绘制哪一类的 SHAP（基于 session 的预测类）
    cls = st.session_state.predicted_class if st.session_state.predicted_class is not None else 0

    # 现在规范出 sample_shap (1D array of feature contributions) 和 expected_value
    sample_shap = None
    expected_value = None

    # 情况 1: shap_values 是 list（多分类情形常见）
    if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and shap_values.dtype == 'object'):
        # 尝试按类索引
        try:
            # shap_values[class] 应为形状 (n_samples, n_features)
            arr = shap_values[cls]
            arr = np.asarray(arr)
            # 取第一个样本
            sample_shap = arr.reshape(arr.shape[0], -1)[0]
            # expected_value 可能是列表或 ndarray
            try:
                expected_value = explainer.expected_value[cls]
            except Exception:
                # shap.Explainer 返回 base_values
                try:
                    expected_value = explainer(base_values=True).base_values[cls]
                except Exception:
                    expected_value = None
        except Exception:
            # 退而求其次：把第一个类/数组转为 sample
            try:
                arr0 = np.asarray(shap_values[0])
                sample_shap = arr0.reshape(arr0.shape[0], -1)[0]
                try:
                    expected_value = explainer.expected_value[0]
                except Exception:
                    expected_value = None
            except Exception as e:
                print("解析 list 形式 shap_values 失败：", e)
                sample_shap = None

    else:
        # 情况 2: shap_values 是 ndarray，可能形状 (n_samples, n_features) 或 (n_classes, n_samples, n_features)
        arr = np.asarray(shap_values)
        if arr.ndim == 2:
            # (n_samples, n_features)
            sample_shap = arr[0, :]
            try:
                expected_value = explainer.expected_value
            except Exception:
                expected_value = None
        elif arr.ndim == 3:
            # (n_classes, n_samples, n_features)
            try:
                sample_shap = arr[cls, 0, :]
                expected_value = explainer.expected_value[cls] if hasattr(explainer, "expected_value") else None
            except Exception:
                # fallback
                sample_shap = arr[0, 0, :]
                expected_value = None
        else:
            # 其它不可预期的形状
            try:
                sample_shap = arr.squeeze()
            except Exception:
                sample_shap = None

    # 最终检查 sample_shap
    if sample_shap is None:
        print("无法从 shap_values 中提取单样本 shap 值，放弃绘图。")
        return False

    # 绘图：优先使用 shap.force_plot(matplotlib=True)，失败则回退
    try:
        plt.figure(figsize=(10, 4))
        # shap.force_plot 需要 expected_value（标量或与 sample_shap 对应）
        try:
            shap.force_plot(expected_value, sample_shap, X_df.iloc[0], matplotlib=True, show=False)
        except Exception:
            # 某些版本需要把 expected_value 转为标量
            try:
                ev = expected_value if expected_value is not None else None
                shap.force_plot(ev, sample_shap, X_df.iloc[0], matplotlib=True, show=False)
            except Exception as e:
                # 回退到 waterfall / bar
                raise e
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close()
        return True

    except Exception as e_force:
        # 回退绘图：用 shap.plots._waterfall.waterfall_legacy 或简单 bar
        try:
            # waterfall（需要 shap >= 0.39），我们尝试通过 shap.plots.waterfall
            plt.figure(figsize=(8, 5))
            # 取绝对值并排序后绘制水平条形图，作为安全回退方案
            vals = np.array(sample_shap)
            order = np.argsort(np.abs(vals))[::-1]
            top_idx = order  # 可限制 top-k
            names = np.array(feature_names)[top_idx]
            vals_sorted = vals[top_idx]
            y_pos = np.arange(len(vals_sorted))
            plt.barh(y_pos, vals_sorted)
            plt.yticks(y_pos, names)
            plt.gca().invert_yaxis()
            plt.title("SHAP 值 (回退条形图)")
            plt.xlabel("SHAP 值")
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight", dpi=300)
            plt.close()
            return True
        except Exception as e_bar:
            print("SHAP 回退绘图失败:", e_force, e_bar)
            return False

# -------------------- 显示结果与解释 --------------------
if st.session_state.prediction_made:
    st.subheader("预测结果")
    class_label = "患病 (1)" if st.session_state.predicted_class == 1 else "未患病 (0)"
    st.write(f"**预测类别:** {class_label}")
    st.write(f"**预测概率:** {st.session_state.predicted_proba}")
    st.write(st.session_state.advice)

    # 生成并显示 SHAP 图
    st.subheader("SHAP 力解释图（自动适配）")
    if not st.session_state.shap_plot_generated:
        ok = generate_shap_plot_and_save(
            model,
            st.session_state.feature_values,
            feature_names,
            out_path="shap_force_plot.png",
        )
        st.session_state.shap_plot_generated = ok

    if st.session_state.shap_plot_generated and os.path.exists("shap_force_plot.png"):
        st.image("shap_force_plot.png", caption="SHAP 力解释图")
    else:
        st.write("无法生成 SHAP 图（请检查模型与 shap 版本）。")

    # 生成 LIME 解释（若有 X_test）
    st.subheader("LIME 解释")
    if X_test is None:
        st.write("缺少 X_test.csv，无法运行 LIME。")
    else:
        try:
            lime_explainer = LimeTabularExplainer(
                training_data=X_test.values,
                feature_names=X_test.columns.tolist(),
                class_names=["未患病", "患病"],
                mode="classification",
            )
            lime_exp = lime_explainer.explain_instance(
                data_row=np.array(st.session_state.features).flatten(),
                predict_fn=model.predict_proba,
            )
            lime_html = lime_exp.as_html(show_table=False)
            st.components.v1.html(lime_html, height=800, scrolling=True)
        except Exception as e:
            st.write("LIME 解释失败：", e)

# -------------------- 结束 --------------------
st.write("提示：修改输入后，直接点击 Predict 会覆盖上一次的结果并刷新 SHAP/LIME 图。")
