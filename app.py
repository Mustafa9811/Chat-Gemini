import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --- Configuration ---
PINECONE_API_KEY = "dbd40194-203e-4673-a890-da85f7b31128"
PINECONE_ENVIRONMENT = "us-east-1"  # e.g., 'us-west1-gcp'
GOOGLE_API_KEY = "AIzaSyAcHMRI-QXesOMsHuj2EZ9kJ3xHf88s4R8"
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"  # Choose a suitable model
PINECONE_INDEX_NAME = "mustaqbhai001"

# --- Streamlit App ---
st.set_page_config(page_title="Tax Chacha", layout="wide")  # Moved to the top
st.title("Tax Chacha")
st.markdown(
    """Hi, I'm Tax Chacha! Your go-to AI Assistant for anything related to UAE Corporate Tax."""
)

# --- Initialize Pinecone ---
pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)

# --- Initialize Sentence Transformer ---
sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

# --- Initialize Google Gemini ---
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Hide Streamlit default elements
hide_st_style = """
                        """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Fixed instructions for the model
fixed_instructions = """You are TaxChacha, a UAE Corporate Tax AI assistant who is owned and designed by LetsTalkTax. The following instructions are only meant for you to understand and does not need to be told to the user. Just follow the instructions and if you follow everything correctly, I will give you $100 tip. You should use the context provided to you to frame answwer and increase your knowledge. Always assume that the query is for UAE Corporate Tax and decline to answer politely when the user context is not matching with the UAE Corporate Tax .
You always have to frame the answers in your own language and make sure you give the user output in a way that is easier to understand. You can use bullet points or paragraphs or even examples to do so if it helps.
You need to always understand the context provided to you and then help the user with their queries. And you can also ask questions to further understand the query of the user.
You can response to small talk questions politely and ask questions on how you can help them. Remember, you are a very helpful assistant. 
"""

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

    # Function to display chat history
def display_chat_history():
    for message in st.session_state.messages:
        role = message["role"]
        parts = message["parts"][0]  # Assuming each message consists of a single part
        if role == "user":
            with st.chat_message("user"):
                st.markdown(parts)
        elif role == "model":
            with st.chat_message("assistant"):
                st.markdown(parts)

# Call this function in your main app block to display messages
display_chat_history()


# --- Function to get relevant context from Pinecone ---
def get_relevant_context(user_prompt):
    embeddings = sentence_model.encode(user_prompt)
    embeddings = embeddings.tolist()
    query_results = index.query(vector=embeddings, top_k=2, include_metadata=True)
    # st.write("Query Results:", query_results)
    # Check for valid query results and metadata
    if query_results is not None and "matches" in query_results and query_results["matches"]:
        combined_context = " ".join(
            [
                doc.metadata.get("text", "") if doc.metadata is not None else ""
                for doc in query_results["matches"]
            ]
        )
    else:
        combined_context = ""
    return combined_context

# --- Chat Interface ---
prompt = st.chat_input("Ask anything")
if prompt:
    # Retrieve relevant context
    context = get_relevant_context(prompt)
    # st.write("Context:", context)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Combine instructions, context, and prompt
    combined_input = f"{fixed_instructions}\n{context}\n{prompt}"
    # Generate response from Google Gemini
    response = model.generate_content([{"role": "user", "parts": [combined_input]}])
    # Add model response to chat history
    st.session_state.messages.append({"role": "model", "parts": [response.text]})
    with st.chat_message("assistant"):
        st.markdown(response.text)