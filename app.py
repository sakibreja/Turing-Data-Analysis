import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# Title and description
st.title("Turing Data Analysis")
st.markdown(
    "This app solves the Turing Data Analysis Quiz, which covers Python, SQL, Data Analysis, and Machine Learning. Upload datasets to begin."
)

# File upload
st.sidebar.header("Upload Your Datasets")
cardio_file = st.sidebar.file_uploader("Cardiovascular Data (cardio_base.csv)", type="csv")
alco_file = st.sidebar.file_uploader("Alcohol Data (cardio_alco.csv)", type="csv")
covid_file = st.sidebar.file_uploader("COVID-19 Data (covid_data.csv)", type="csv")

if cardio_file and alco_file and covid_file:
    # Load datasets
    base = pd.read_csv(cardio_file)
    alco = pd.read_csv(alco_file, sep=';')
    covid = pd.read_csv(covid_file)

    st.success("Datasets uploaded successfully!")

    # Q1: Age group with highest and lowest average weight
    base['age'] = base['age'] // 365
    base['age'] = base['age'].astype(int)

    q1 = base[['age', 'weight']].groupby('age').mean()
    q1 = q1.sort_values('weight', ascending=False).reset_index()
    max_weight = q1['weight'].max()
    min_weight = q1['weight'].min()
    diff = max_weight - min_weight
    diff_percent = (diff / min_weight) * 100

    st.subheader("Q1: Age Group Weight Analysis")
    st.write(f"The absolute weight difference is {diff:.2f}kg, which is {diff_percent:.2f}% higher.")

    # Q2: Cholesterol levels for 50+
    q2 = base[['age', 'cholesterol']].groupby('age').mean()
    q2['50+'] = q2.index > 50
    q2 = q2[['50+', 'cholesterol']].groupby('50+').mean().reset_index()
    diff_percent = ((q2.iloc[1, 1] / q2.iloc[0, 1]) - 1) * 100

    st.subheader("Q2: Cholesterol Levels by Age")
    st.write(f"Cholesterol levels for 50+ are {diff_percent:.2f}% higher than younger individuals.")

    # Q3: Smoking likelihood by gender
    q3 = base[['gender', 'smoke']].groupby('gender').sum().reset_index()
    diff = q3.iloc[1, 1] / q3.iloc[0, 1]

    st.subheader("Q3: Smoking Likelihood by Gender")
    st.write(f"Men are {round(diff)} times more likely to smoke than women.")

    # Q4: Tallest 1%
    tallest_1_percent = np.percentile(base['height'], 99)

    st.subheader("Q4: Tallest 1%")
    st.write(f"The tallest 1% of people are taller than {tallest_1_percent:.2f} cm.")

    # Q5: Spearman rank correlation
    spearman_correlation = base.corr(method='spearman')
    corr_pairs = spearman_correlation.unstack().sort_values(ascending=False)
    highest_corr = corr_pairs[(corr_pairs != 1.0)].reset_index().iloc[0]

    st.subheader("Q5: Highest Spearman Correlation")
    st.write(
        f"The highest Spearman correlation is between {highest_corr['level_0']} and {highest_corr['level_1']} with a value of {highest_corr[0]:.2f}."
    )

    # Q6: Height standard deviation
    avg_height = base['height'].mean()
    std_dev = base['height'].std()
    base['is_far'] = abs(base['height'] - avg_height) > (2 * std_dev)
    far_percentage = (base['is_far'].mean()) * 100

    st.subheader("Q6: Height Deviation Analysis")
    st.write(f"{far_percentage:.2f}% of people are more than 2 standard deviations away from the average height.")

    # Q7: Alcohol consumption for 50+
    q7 = pd.merge(base, alco, on='id', how='left')
    q7 = q7[~q7['alco'].isna()]
    q7['50+'] = q7['age'] > 50
    q7 = q7[q7['50+']]
    alco_percentage = (q7['alco'].mean()) * 100

    st.subheader("Q7: Alcohol Consumption for 50+")
    st.write(f"{alco_percentage:.2f}% of individuals over 50 consume alcohol.")

    # Further questions...
    st.info("Additional analysis can be implemented as needed.")
else:
    st.warning("Please upload all required datasets to start analysis.")
