from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv



load_dotenv()

st.header('Reasearch Tool')

user_input = st.text_input("Enter your research query:")


if st.button('Submit'):
    result = model.invoke(user_input)
    st.write("Processing...")