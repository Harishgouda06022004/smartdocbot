from data_loader import load_and_split_pdf
from vector_store import create_vector_store
from langchain.chains import RetrievalQA
from chain import llm  # or ChatGroq
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    pdf_path = "data/3rdSemSyllabus.pdf"

    # Load and split
    chunks = load_and_split_pdf(pdf_path)

    # Create vector store
    vectordb = create_vector_store(chunks)

    # Create retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,
        chain_type="stuff"
    )

    print("🤖 SmartDocBot is ready! Ask anything about your document.\n")

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        result = qa_chain.run(query)
        print("Bot:", result)
