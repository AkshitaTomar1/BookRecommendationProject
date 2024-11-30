import pickle 
import streamlit as st
import numpy as np

st.header("Books recommendation System using Machine Learning")
model=pickle.load(open('artifacts/model.pkl','rb'))
book_names=pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating=pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot=pickle.load(open('artifacts/book_pivot.pkl','rb'))



selected_books=st.selectbox("Type or select a book",)