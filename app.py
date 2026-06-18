import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Movie Popularity Predictor", page_icon="🎬", layout="centered")

@st.cache_resource
def load_models():
    try:
        # Load your model, vectorizer, and scaler using the exact filenames
        model = joblib.load('best_model.pkl')
        with open('vectorizer.pkl', 'rb') as f:
            cv = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, cv, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, cv, scaler = load_models()

st.title("🎬 Movie Popularity Predictor")
st.write("This app predicts the **Popularity Class** of a movie.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    vote_average = st.number_input("Vote Average", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    vote_count = st.number_input("Total Vote Count", min_value=0, max_value=50000, value=2148, step=10)

with col2:
    release_month = st.slider("Release Month", min_value=1, max_value=12, value=6)
    release_day = st.slider("Release Day", min_value=1, max_value=31, value=15)

st.divider()
movie_title = st.text_input("Movie Title", value="The Shawshank Redemption")
movie_overview = st.text_area("Overview", value="A movie about hope and freedom.")

if st.button("🔮 Predict Popularity", type="primary"):
    if model is not None:
        try:
            # 1. Scale numeric inputs using RobustScaler (expects shape matching training scaling)
            # Your scaler was fit on: ['vote_average', 'vote_count', 'popularity']
            # We provide a dummy 0 for popularity since it's dropped during X/y split anyway
            scaled = scaler.transform([[vote_average, vote_count, 0]])
            vote_average_scaled = scaled[0][0]
            vote_count_scaled = scaled[0][1]

            # 2. Vectorize the overview text feature block (max_features=50)
            overview_cv = cv.transform([movie_overview]).toarray()

            # 3. Create numeric feature array (4 features: vote_average, vote_count, release_month, release_day)
            numeric_features = np.array([[vote_average_scaled, vote_count_scaled, release_month, release_day]])
            
            # 4. Concatenate numeric and text features to match the exact shape expected by the model
            input_data = np.concatenate([numeric_features, overview_cv], axis=1)

            # 5. Predict popularity class
            prediction = model.predict(input_data)[0]

            st.divider()
            st.subheader("🎯 Prediction Results")

            if prediction == 0:
                st.success("🟢 Low Popularity")
                st.write(f"📉 **{movie_title}** is predicted to have **Low** popularity.")
            elif prediction == 1:
                st.warning("🟡 Medium Popularity")
                st.write(f"👍 **{movie_title}** is predicted to have **Medium** popularity.")
            else:
                st.error("🔴 High Popularity")
                st.balloons()
                st.write(f"🌟 **{movie_title}** is predicted to have **High** popularity!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
