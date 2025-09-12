# learning_assistant_prompts.py
# Prompt templates for generating questions and answers using LLMs in the Personal Learning Assistant app.

prompts = {
    "question": """
    You are a helpful teaching assistant. Based on the following study material:

    {text}

    Generate 15 high-quality practice questions that cover key concepts.
    Do not include answers, only the questions.
    """,
    "answer": """
    You are a helpful teaching assistant. Based on the following study material, provide a
    comprehensive answer to the question. If the information is not available in the provided
    material, state that explicitly.

    Study Material:
    {context}

    Question: {question}

    Answer:
    """
}
# The 'prompts' dictionary contains two templates:
# - 'question': Used to instruct the LLM to generate practice questions from study material.
# - 'answer': Used to instruct the LLM to answer a selected question using relevant context.
