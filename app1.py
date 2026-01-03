# this is for streamlit web app
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


# load the trained model
with open('ohe_geo.pkl','rb') as file:
    ohe_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)


# streamlit app
st.title('Customer churn prediction')

# user input
geography=st.selectbox('Geography',ohe_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Memner',[0,1])

# prepare input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([(gender)])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]

})

# # geography col is missing
# geo_encoded=ohe_geo.transform([[geography]]).toarray()
# geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_geo.get_feature_names_out(['Geography']))

# One-hot encode geography (FOR SINGLE INPUT)
geo_encoded = ohe_geo.transform(pd.DataFrame({'Geography': [geography]}))

geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=ohe_geo.get_feature_names_out(['Geography'])
)


# concat
input_data=pd.concat([input_data,geo_encoded_df],axis=1)

# scaled
input_data_scaled=scalar.transform(input_data)

model=load_model('model.h5')
# prediction churn
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

st.write(f'Cuurn Probability: {prediction_proba:.2f}')

if prediction_proba>0.5:
    print("The customer is likely to churn")
else:
    print("The customer is not likely to churn")