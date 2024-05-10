import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
import PIL 
import io
#from st_paywall import add_auth

st.set_page_config(page_title="Tax Chacha", layout = 'wide')

st.title('Tax Chacha')
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown("""
Hi, I'm Tax Chacha! Your go-to AI Assistant for anything related to UAE Corporate Tax.
""")

# add_auth(required=True)

# st.write(f"Subscription Status: {st.session_state.user_subscribed}")
# st.write("Congratulations! You're all set and subscribed!")
# st.write(f"By the way, your email is: {st.session_state.email}")

GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

genai.configure(api_key=GOOGLE_API_KEY)

# model config
generation_config = {
"temperature": 0.1,
"top_p": 1,
"top_k": 1,
"max_output_tokens": 1048,
}

safety_settings = [
{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
},
{
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
},
{
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
},
{
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
},
]

# Using "with" notation for the sidebar
with st.sidebar:
    st.title('Input Type:')
    # Now, we only allow text input
    add_radio = st.radio(
        "Choose your input type:",
        ("Text Input ✏️",),  # Single option now
    )

# Fixed instructions for the model
fixed_instructions = """
# Add your fixed instructions here
"""

if "model_messages" not in st.session_state:
    st.session_state.model_messages = []

# Initialize previous_input_type in session_state if it doesn't exist
if "previous_input_type" not in st.session_state:
    st.session_state.previous_input_type = None

# Check if the input type has changed
#if st.session_state.previous_input_type != add_radio:
#    # Clear the messages
#    st.session_state.messages = []
#    # Update previous_input_type
#    st.session_state.previous_input_type = add_radio

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"][0])

if add_radio == 'Text Input ✏️':
    model = genai.GenerativeModel(model_name="gemini-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    prompt = st.chat_input("Ask anything")

    if prompt:
        # Add user's prompt to the visible messages
        st.session_state.messages.append({
            "role": "user",
            "parts": [prompt]
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Combine the fixed instructions with the prompt for the model's input
        combined_input = fixed_instructions + prompt
        st.session_state.model_messages.append({
            "role": "user",
            "parts": [combined_input]
        })

        # Generate response from the model based on the combined input
        response = model.generate_content(st.session_state.model_messages)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response.text)

        # Add the model's response to both message histories
        st.session_state.messages.append({
            "role": "model",
            "parts": [response.text]
        })
        st.session_state.model_messages.append({
            "role": "model",
            "parts": [response.text]
        })
