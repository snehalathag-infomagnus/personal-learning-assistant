# 1. Imports
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, add_messages, START
from langchain.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from datetime import datetime
 
load_dotenv()
 
# 2. Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
 

# Each person is a dict with id, name, age, dob
PLS = [
    {"id": "pls1", "name": "peter", "age": 30, "dob": "1995-01-01"},
    {"id": "pls2", "name": "paul", "age": 28, "dob": "1997-02-15"},
    {"id": "pls3", "name": "mary", "age": 32, "dob": "1993-03-10"},
    {"id": "pls4", "name": "alice wonder", "age": 27, "dob": "1998-04-20"},
    {"id": "pls5", "name": "john smith", "age": 35, "dob": "1990-05-05"},
]

CMP = [
    {"id": "cmp1", "name": "steve", "age": 40, "dob": "1985-06-12"},
    {"id": "cmp2", "name": "paul john", "age": 29, "dob": "1996-07-23"},
    {"id": "cmp3", "name": "mary jane", "age": 31, "dob": "1994-08-30"},
    {"id": "cmp4", "name": "john doe", "age": 33, "dob": "1992-09-17"},
    {"id": "cmp5", "name": "alice smith", "age": 26, "dob": "1999-10-11"},
]
 

@tool
def pls(name: str) -> str:
    """Return all PLS persons whose name contains the input as a substring (case-insensitive), with all fields."""
    name_lower = name.lower()
    matches = [p for p in PLS if name_lower in p["name"].lower()] or [p for p in PLS if name_lower in p["id"].lower()]
    if matches:
        return "PLS account(s) found:\n" + "\n".join(
            [f"id: {p['id']}, name: {p['name']}, age: {p['age']}, dob: {p['dob']}" for p in matches]
        )
    else:
        return "No matching PLS account found."


@tool
def cmp(name: str) -> str:
    """Return all CMP persons whose name contains the input as a substring (case-insensitive), with all fields."""
    name_lower = name.lower()
    matches = [p for p in CMP if name_lower in p["name"].lower()] or [p for p in CMP if name_lower in p["id"].lower()]
    if matches:
        return "CMP account(s) found:\n" + "\n".join(
            [f"id: {p['id']}, name: {p['name']}, age: {p['age']}, dob: {p['dob']}" for p in matches]
        )
    else:
        return "No matching CMP account found."

@tool
def classify_name(name: str) -> str:
    """Classify a name as 'PLS', 'CMP', or 'Unknown' using substring match."""
    if pls(name):
        return "PLS"
    elif cmp(name):
        return "CMP"
    else:
        return "Unknown"
 
@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
 
# 4. Wrap the tool with ToolNode
tool_node = ToolNode([current_datetime, calculator, pls, cmp])
 
tools=[current_datetime, calculator, pls, cmp]
# Initialize the model
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    temperature=0.5  
)
    # bind tools to the model so it knows about calculator & search_web
    # Syntax might differ; current versions use model.bind_tools()
model_with_tools = model.bind_tools(tools)
   

def call_model(state: State):
    # System prompt for the LLM
    system_prompt = (
        "You have access to two types of accounts: PLS and CMP. "
        "If a user query matches more than one record in both PLS and CMP accounts, do not display all the records. "
        "Instead, respond: 'Multiple records found in both PLS and CMP accounts. Please specify the exact person (name or id) you want information for.' "
        "Only provide detailed information when the user clearly specifies the account type or the exact person."
    )
    # Insert system prompt as the first message if not already present
    messages = state["messages"]
    if not (messages and getattr(messages[0], "role", None) == "system"):
        from langchain.schema import SystemMessage
        messages = [SystemMessage(content=system_prompt)] + messages
    resp = model_with_tools.invoke(messages)
    return {"messages": state["messages"] + [resp]}
 
# 5. Build the graph
builder = StateGraph(State)
 
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START,"model")
builder.add_conditional_edges("model",tools_condition,{"tools": "tools", END: END})
builder.add_edge("tools", "model")
builder.set_entry_point("model")
# Creates a runnable graph object
graph = builder.compile()
 
def main():
    print(" Bot (type 'exit' to quit)")
    history = []
 
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Goodbye!")
            print("User question and bot answer history:")
            for i in range(0, len(history), 2):
                user_msg = history[i].content if i < len(history) else ""
                bot_msg = history[i+1].content if i+1 < len(history) else ""
                print(f"  {i//2 + 1}. User: {user_msg} | Bot: {bot_msg}")
            break
 
        # Add user message to history
        history.append(HumanMessage(content=user_input))
        # Define the state with current messages
        state = {"messages": history}
 
        # Run the graph
        result = graph.invoke(state)
 
        # Get and print bot response
        bot_reply = result["messages"][-1].content
        print(f"Bot: {bot_reply}")
 
        # Update history with bot response
        history.append(result["messages"][-1])
 
# 7. Run the chatbot
if __name__ == "__main__":
    main()