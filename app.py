import streamlit as st
import pandas as pd
import pickle
import numpy as np


with open('ohe_company.pkl','rb') as file:
    ohe_company = pickle.load(file)

with open('ohe_fuel_type.pkl','rb') as file:
    ohe_fuel = pickle.load(file)

with open('ohe_name.pkl','rb') as file:
    ohe_name = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


with open('linear.pkl','rb') as file:
    linear = pickle.load(file)
    
    

st.title('Car Prices Predictions')


company_st = st.selectbox('Select the company of the car',ohe_company.categories_[0])

year = st.number_input('Enter the purchasing year')
kms = st.number_input('Enter the Kms_driven')

fuel_type_st = st.selectbox('Select the Fuel type',ohe_fuel.categories_[0])

new_name_st = st.selectbox('Select the car name ',ohe_name.categories_[0])


input_data={

    #'company':[company_st],
    'year':year,
    'kms_driven':kms
    #'fuel_type':[fuel_type_st],
    #'new_name':[new_name_st]
}



input_data_df = pd.DataFrame([input_data])

# now using one hot encoding on the company
company_one = ohe_company.transform([[company_st]]).toarray()
company_df = pd.DataFrame(company_one,columns=ohe_company.get_feature_names_out())

# now using one hot encoding on the fuel type
fuel_one = ohe_fuel.transform([[fuel_type_st]]).toarray()
fuel_df = pd.DataFrame(fuel_one,columns=ohe_fuel.get_feature_names_out())

# now using one hot encoding on the new_name
new_name_one = ohe_name.transform([[new_name_st]]).toarray()
new_name_df = pd.DataFrame(new_name_one,columns=ohe_name.get_feature_names_out())


df = pd.concat([input_data_df.reset_index(drop=True),new_name_df,company_df,fuel_df],axis=1)

df_scaler = scaler.transform(df)

rand_prob = linear.predict(df_scaler)
prob = rand_prob[0]

if st.button('Predict'):
    st.title(prob)
    
