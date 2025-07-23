from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

qa_chain = load_qa_chain(llm, chain_type="stuff")