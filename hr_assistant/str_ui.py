import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from llm import response_llm

st.title('HR Assistant : RAG-Powered')

user_query = st.chat_input("Enter your Query")

# session state
if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [AIMessage(content='Hello, How can I help you?')]



# conversation

for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message('AI'):
            st.write(message.content)
        #st.session_state.chat_history.append(message)

    elif isinstance(message,HumanMessage):
        with st.chat_message('Human'):
            st.write(message.content)

# user input

if user_query is not None and user_query != "":
    with st.chat_message('Human'):
        st.markdown(user_query)

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message('AI'):
        response, sources = response_llm(
            user_query=user_query,
            chat_history=st.session_state.chat_history
        )

        st.write(response)
        st.markdown("---")
        st.markdown("### Sources used")

        for i, source in enumerate(sources):
            with st.expander(f"Chunk {i + 1}"):
                st.write(source["content"])
                st.caption(source["metadata"])
    st.session_state.chat_history.append(AIMessage(content=response))