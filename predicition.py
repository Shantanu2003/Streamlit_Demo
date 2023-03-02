# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
from sklearn.preprocessing import PolynomialFeatures 

# Load the saved model
loaded_model= pickle.load(open('D:/SHANTANU/InternFastFind/Major Project/model.sav','rb'))

# Paste the training code

poly = PolynomialFeatures(degree=2)
years=10
print(f'Prediction -Morality Rate Of World {2020.5+years} will be:',end=' ')
print(loaded_model.predict(poly.fit_transform([[2020.5+years]])))