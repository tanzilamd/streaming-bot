import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


# App config

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Tanzil Bot", page_icon="🤖")

st.title("Tanzilbot")


# get response

def get_response(query, chat_history):
    template = """
        You are Tanzilbot, created by Tanzil Ahmed, a high school student from Dhaka, Bangladesh.
        You help students with topics such as university applications, scholarships, visa processes, cost of living, Exam requirements like IELTS, TOEFL, SAT, GRE and adjusting to a new country.
        Help users with study abroad topics like:

        Keep responses clear, helpful, and concise. Provide actionable advice, useful resources, and use Markdown & bullet points for better clarity.

        Chat history: {chat_history}
        User's question: {user_question}
        
        Now, provide structured response.
    """

    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOpenAI()
    llm = ChatGroq(model="llama3-8b-8192")

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })



# conversation

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


# user input

user_query = st.chat_input("Your message")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))


    st.session_state.chat_history.append(AIMessage(ai_response))
