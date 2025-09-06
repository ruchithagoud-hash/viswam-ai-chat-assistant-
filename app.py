import streamlit as st
import openai

# Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🤖 Viswam AI Chat Assistant")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from OpenAI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # you can use "gpt-4" if available
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"]
                ]
            )
            reply = response["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"⚠️ Error: {e}"

        message_placeholder.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
