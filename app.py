import os
import re
import docx2txt
import pandas as pd
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
from youtube_transcript_api import YouTubeTranscriptApi 
import urllib.request
from bs4 import BeautifulSoup
import html2text
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

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


def process_documents(files):
    extracted_texts = []
    
    for file in files:
        text = process_document(file)
        if text is not None:
            extracted_texts.append(text)
    
    return ' '.join(extracted_texts)


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
    
def get_youtube_video_id(url):
    # Expresión regular para buscar el ID del video en la URL de YouTube
    pattern = r'(?:youtu\.be/|youtube\.com/watch\?v=|youtube\.com/embed/|youtube\.com/v/|youtube\.com/.*[?&]v=|youtube\.com/.*embed/|youtube\.com/.*v/|youtube\.com/embed/|youtube\.com/.*[?&]v=|youtube\.com/.*[?&]vi=|youtube\.com/.*[?&]v=|embed/\?videoId=|embed/iframe/|.*be\.com.*[?]v=|.*be\.com/.*[?]v=)([^"&?/ ]{11})'

    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    else:
        return None
    
def process_links(links):
    extracted_texts = []
    
    for link in links:
        text = process_link(link)
        if text is not None:
            extracted_texts.append(text)
    
    return ' '.join(extracted_texts)

def process_link(link):
    text = ""

    if "youtube.com" in link:
        video_id = get_youtube_video_id(link)
        try:
            srt = YouTubeTranscriptApi.get_transcript(video_id, ('es','en'))
            for item in srt:
                text += item['text'] + ' '
        except Exception as e:
            print(f"Error al obtener los subtítulos: {e}")
            text += 'No se ha podido cargar la informacion del video'

    else:
        uf = urllib.request.urlopen("https://businessinsider.mx/como-iniciar-propio-negocio-como-renunciar-a-un-trabajo/")
        html = uf.read()

        soup = BeautifulSoup(html, 'html.parser')

        page_text = soup.get_text()

        text_converter = html2text.HTML2Text()
        text += text_converter.handle(page_text)

    print(text)
    return text

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_text_from_request(request):
    text = "."
    documents = request.files.getlist('documents[]')
    links = request.form.getlist('links[]')

    if documents:
        text += process_documents(documents)
    if links:
        text += process_links(links)

    return text


def create_embeddings(chunks):
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

@app.route('/', methods=['POST'])
def documents():
    if request.method == 'POST':
        user_question = request.form['chat']

        text = get_text_from_request(request)
        chunks = split_text_into_chunks(text)

        knowledge_base = create_embeddings(chunks)

        if user_question:
            response = handle_user_question(knowledge_base, user_question)
            return response
        

def save_chat_history(chat_key, text):
    global chat_history
    if chat_key not in chat_history:
        chat_history[chat_key] = []
    chat_history[chat_key].append(text)

def get_chat_history(chat_key):
    global chat_history

    if chat_key in chat_history:
        return chat_history[chat_key]
    else:
        return []

chat_history = {}

@app.route('/raw', methods=['POST'])
def raw():
    global chat_history
    if request.method == 'POST':
        user_question = request.form['question']
        chat_key = request.form['chatKey']
        
        # Contexto del asistente
        context = {"role": "system", "content": "Eres un asistente virtual."}
        messages = [context]

        # agregar documentos, links, etc
        text = get_text_from_request(request)
        messages.append({"role": "user", "content": text})

        # Obtiene la conversación anterior o crea una nueva
        chat_history[chat_key] = get_chat_history(chat_key)

        for chat_text in chat_history[chat_key]:
            messages.append({"role": "user", "content": chat_text})

        # Agrega la pregunta del usuario actual
        messages.append({"role": "user", "content": user_question})

        print(messages)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)

        response_content = response.choices[0].message['content']

        # Almacena la conversación actual
        save_chat_history(chat_key, user_question)

        return response_content


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))