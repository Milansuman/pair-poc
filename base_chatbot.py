from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(
    model="llama2"
)

def generate_response(prompt, system_prompt="You are a helpful chatbot named Marvin."):
    prompt_template = ChatPromptTemplate.from_template("""
[INST]{system} Answer the prompt in a simple sentence.
{prompt}
[/INST]""")

    chain = prompt_template | llm | StrOutputParser()

    response = chain.invoke({
        "system": system_prompt,
        "prompt": prompt
    })

    return response
