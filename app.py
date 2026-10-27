"""Streamlit demo for the Bank Marketing classification case study.
Run: streamlit run app.py
Train/save the selected model first with: python analyze.py
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'best_bank_marketing_model.joblib'

st.set_page_config(page_title='Bank Marketing Predictor', page_icon='🏦', layout='centered')
st.title('Bank Term-Deposit Subscription Predictor')
st.write('Estimate the model’s probability that a customer will subscribe to a term deposit. This classroom demo uses the selected Decision Tree pipeline trained on the supplied bank-full dataset.')
st.info('The model excludes call duration, because duration is not known before a call. Predictions are exploratory and should not be used as the sole basis for a real banking decision.')

if not MODEL_PATH.exists():
    st.error('Trained model not found. From the project folder, run `python analyze.py` first.')
    st.stop()

model = joblib.load(MODEL_PATH)
with st.form('customer_input'):
    st.subheader('Customer and campaign details')
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider('Age', 18, 95, 35)
        job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox('Marital status', ['married', 'single', 'divorced'])
        education = st.selectbox('Education', ['secondary', 'tertiary', 'primary', 'unknown'])
        default = st.selectbox('Credit in default?', ['no', 'yes'])
        balance = st.number_input('Average yearly balance (euros, dataset units)', min_value=-10000, max_value=1000000, value=1000, step=100)
        housing = st.selectbox('Housing loan?', ['yes', 'no'])
        loan = st.selectbox('Personal loan?', ['no', 'yes'])
    with c2:
        contact = st.selectbox('Contact communication type', ['cellular', 'telephone', 'unknown'])
        day = st.slider('Last contact day of month', 1, 31, 15)
        month = st.selectbox('Last contact month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        campaign = st.number_input('Contacts in current campaign', min_value=1, max_value=100, value=1, step=1)
        pdays = st.number_input('Days since previous contact (-1 = not previously contacted)', min_value=-1, max_value=999, value=-1, step=1)
        previous = st.number_input('Number of contacts before this campaign', min_value=0, max_value=100, value=0, step=1)
        poutcome = st.selectbox('Outcome of previous campaign', ['unknown', 'failure', 'success', 'other'])
    submitted = st.form_submit_button('Estimate subscription probability', type='primary')

if submitted:
    customer = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
    }])
    probability = float(model.predict_proba(customer)[0, 1])
    predicted = int(probability >= 0.5)
    st.metric('Estimated subscription probability', f'{probability:.1%}')
    st.progress(probability)
    if predicted:
        st.success('Model classification at the default 0.50 threshold: likely to subscribe (yes).')
    else:
        st.warning('Model classification at the default 0.50 threshold: not predicted to subscribe (no).')
    st.caption('This probability is not a guarantee. The selected model’s test F1-score was approximately 0.345, and its recall for subscribers was approximately 29.3%.')
