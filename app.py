from flask import Flask, render_template, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from flask_cors import CORS
import docx2txt
import pandas as pd


app = Flask(__name__)
CORS(app)

knowledge_bases = {}

def excel_to_text(input_excel_path):
    try:
        # Read the Excel file
        df = pd.read_excel(input_excel_path)

        # Get the columns and their names
        columns = df.columns.tolist()

        # Convert the DataFrame to plain text with custom formatting
        text = ''
        text += '|'.join(columns) + '\n'  # Column headers separated by '|'
        text += '|'.join(['_' * len(col) for col in columns]) + '\n'  # Separation line

        # Iterate over the rows and add the content to plain text
        for _, row in df.iterrows():
            text += '|'.join(str(val) for val in row) + '\n'

        return text
    except Exception as e:
        print('Error:', str(e))


def process_document(file):
    if file.filename.endswith('.pdf'):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file.filename.endswith('.docx'):
        text = docx2txt.process(file)
        return text
    elif file.filename.endswith('.xlsx'):
        text = excel_to_text(file)
        return text
    else:
        return None


def process_documents(files):
    extracted_texts = []
    
    for file in files:
        text = process_document(file)
        if text is not None:
            extracted_texts.append(text)
    
    return ' '.join(extracted_texts)

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_embeddings(chunks):
    global knowledge_bases
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

def handle_user_question(knowledge_base, user_question):
    docs = knowledge_base.similarity_search(user_question)
    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
        print(response)
    
    return response

@app.route('/', methods=['GET', 'POST'])
def documents():
    if request.method == 'GET':
        return 'hello world'
    if request.method == 'POST':
        documents = request.files.getlist('documents[]')

        file_key = request.form['filekey']
        user_question = request.form['chat']

        if documents:

            if file_key in knowledge_bases:
                knowledge_base = knowledge_bases.get(file_key)
            else:
                text = process_documents(documents)
                chunks = split_text_into_chunks(text)

                knowledge_base = create_embeddings(chunks)
                knowledge_bases[file_key] = knowledge_base

            if user_question:
                response = handle_user_question(knowledge_base, user_question)
                return response

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))