from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re

load_dotenv()

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

llm = Ollama(
    model="llama3"
)

def judge_response(prompt, response):
    prompt_template = ChatPromptTemplate.from_template("""
You are an ai that judges a response based on the given prompt. Assign a score from 0 to 1 based on how closely the response complies with the prompt. Generate only the score as a number.
Prompt: {prompt}
Response: {response}                                                                                                                                                           
""")
    
    chain = prompt_template | llm | StrOutputParser()

    result = chain.invoke({
        "prompt": prompt,
        "response": response
    })

    return float(re.findall("\d+\.?\d*", result)[0])