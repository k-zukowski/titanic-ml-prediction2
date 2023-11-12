import streamlit as st
import joblib
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model2.sv"
model = joblib.load(filename)

label_mappings = {
    "ChestPainType": {'angina': 0, 'asymptomatic': 1, 'non-anginal pain': 2, 'typical angina': 3},
    "Sex": {'female': 0, 'male': 1},
    "RestingECG": {'hypertrophy': 0, 'normal': 1, 'ST-T wave abnormality': 2},
    "ExerciseAngina": {'no': 0, 'yes': 1},
    "ST_Slope": {'downsloping': 0, 'flat': 1, 'upsloping': 2}
}

def main():
    st.set_page_config(page_title="Heart Disease Prediction")
    st.image("https://cdn.discordapp.com/attachments/330278121104343041/1170772539736342619/IMG_0165.png?ex=655a41e5&is=6547cce5&hm=78411c5b49268a538bab96e6d473e723b9e5931521db24c336c2aa7bf94f0852&")


    st.title("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age_slider = st.slider("Age", value=50, min_value=29, max_value=77)
        sex_radio = st.radio("Sex", list(label_mappings["Sex"].keys()), format_func=lambda x: x)
        chest_pain_radio = st.radio("Chest Pain Type", list(label_mappings["ChestPainType"].keys()),
                                    format_func=lambda x: x)
        resting_bp_slider = st.slider("Resting Blood Pressure", value=128, min_value=94, max_value=200)
        cholesterol_slider = st.slider("Cholesterol", value=246, min_value=126, max_value=564)
        fasting_bs_radio = st.radio("Fasting Blood Sugar", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        rest_ecg_radio = st.radio("Resting ECG", list(label_mappings["RestingECG"].keys()), format_func=lambda x: x)
        max_hr_slider = st.slider("Max Heart Rate", value=149, min_value=71, max_value=202)
        ex_angina_radio = st.radio("Exercise-Induced Angina", list(label_mappings["ExerciseAngina"].keys()),
                                   format_func=lambda x: x)
        oldpeak_slider = st.slider("Oldpeak", value=1.04, min_value=0.0, max_value=6.2)
        st_slope_radio = st.radio("ST Slope", list(label_mappings["ST_Slope"].keys()), format_func=lambda x: x)

    data_for_prediction = [age_slider, label_mappings["Sex"][sex_radio], label_mappings["ChestPainType"][chest_pain_radio],
                            resting_bp_slider, cholesterol_slider, fasting_bs_radio,
                            label_mappings["RestingECG"][rest_ecg_radio], max_hr_slider,
                            label_mappings["ExerciseAngina"][ex_angina_radio], oldpeak_slider,
                            label_mappings["ST_Slope"][st_slope_radio]]

    prediction = model.predict([data_for_prediction])[0]
    confidence = model.predict_proba([data_for_prediction])[0][prediction]

    st.subheader("Would the person have Heart Disease?")
    st.subheader("Yes" if prediction == 1 else "No")
    st.write("Prediction Confidence: {:.2%}".format(confidence))

if __name__ == "__main__":
    main()