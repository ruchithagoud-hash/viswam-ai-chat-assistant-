import streamlit as st

st.title("Viswam AI Chat Assistant")
st.write("Hello from Viswam AI Assistant 🚀")

user_input = st.text_input("Type your message:")

if user_input:
    st.write(f"You said: {user_input}")
