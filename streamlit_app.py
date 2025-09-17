import streamlit as st
from langchain.schema import HumanMessage
from langgraph_demo import graph

st.set_page_config(page_title="Account Lookup Assistant", page_icon="🤖")
st.title("Account Lookup Assistant 🤖")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

for i in range(0, len(st.session_state["history"]), 2):
    user_msg = st.session_state["history"][i].content if i < len(st.session_state["history"]) else ""
    bot_msg = st.session_state["history"][i+1].content if i+1 < len(st.session_state["history"]) else ""
    st.markdown(f"**User:** {user_msg}")
    if bot_msg:
        st.markdown(f"**Bot:** {bot_msg}")


# Input box and send button at the bottom using a form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter a name or query:", "", key="user_input")
    submitted = st.form_submit_button("Send")
    if submitted and user_input.strip():
        st.session_state["history"].append(HumanMessage(content=user_input))
        state = {"messages": st.session_state["history"]}
        result = graph.invoke(state)
        bot_reply = result["messages"][-1].content
        st.session_state["history"].append(result["messages"][-1])
        st.session_state["last_input"] = user_input
        st.rerun()
