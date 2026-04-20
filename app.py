import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("model.pkl")

st.markdown("""
# 🏠 Housing Price Classifier  
住宅価格分類アプリ  

Predict whether a house is Cheap, Medium, or Expensive  
住宅が安い・中程度・高いかを予測します
""")

# Feature Importance 
feature_names = ["lstat", "rm", "crim", "nox", "indus"]
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

predict_button = st.button("Predict / 予測する")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Input Features / 特徴入力")
    st.caption("Adjust sliders to simulate a house and predict its price category. スライダーを調整して、住宅の特徴をシミュレートし、価格カテゴリを予測します。")
    lstat = st.slider("LSTAT – Lower income population (%) / 低所得者層", 0.0, 40.0, 12.0)
    rm = st.slider("RM – Number of rooms / 部屋数", 3.0, 9.0, 6.0)
    crim = st.slider("CRIM – Crime rate / 犯罪率", 0.0, 100.0, 0.1)
    nox = st.slider("NOX – Pollution level / 汚染レベル", 0.3, 1.0, 0.5)
    indus = st.slider("INDUS – Industrial area (%) / 工業地域比率", 0.0, 30.0, 8.0)

    st.divider()
    st.caption("Built with Machine Learning using Random Forest. ランダムフォレストを使用した機械学習で構築されました。")
    
with col2:
    st.markdown("### Prediction / 予測結果")

    if predict_button:
        input_data = pd.DataFrame([{
            "lstat": lstat,
            "rm": rm,
            "crim": crim,
            "nox": nox,
            "indus": indus
        }])

        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)

        proba_dict = dict(zip(model.classes_, proba[0]))
        confidence = proba_dict[prediction[0]]

        # Prediction Result
        if prediction[0] == "cheap":
            st.success(f"💸 Cheap / 安い ({confidence:.2f})")
        elif prediction[0] == "medium":
            st.warning(f"🏠 Medium / 中程度 ({confidence:.2f})")
        else:
            st.error(f"💎 Expensive / 高い ({confidence:.2f})")

        # Prediction Confidence
        st.divider()
        st.markdown("### How sure is the model?\nどのくらい確信しているか？")
        st.write(f"💸 Cheap: {proba_dict['cheap']:.2f}")
        st.progress(float(proba_dict['cheap']))

        st.write(f"🏠 Medium: {proba_dict['medium']:.2f}")
        st.progress(float(proba_dict['medium']))

        st.write(f"💎 Expensive: {proba_dict['expensive']:.2f}")
        st.progress(float(proba_dict['expensive']))

        st.divider()
        st.subheader("Why this prediction?\n予測の要因")
        st.bar_chart(importance_df.set_index("feature"))
        st.caption("Higher bars indicate features that had more influence on the prediction. より高いバーは、予測により影響を与えた特徴を示しています。")

    else:
        st.info("Click 'Predict' to see results\n「予測する」をクリックしてください")