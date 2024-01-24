import os
import re
import docx2txt
import pandas as pd
from flask import Flask, request
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from flask_cors import CORS, cross_origin
from youtube_transcript_api import YouTubeTranscriptApi 
import urllib.request
from bs4 import BeautifulSoup
import html2text
import openai
from urllib.error import HTTPError
import pandas as pd
from plotai import PlotAI
import uuid

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)

openai.api_key = os.getenv('OPENAI_API_KEY')
development = os.getenv('ENV') == 'development'

app = Flask(__name__, static_url_path='', static_folder=ROOT_DIR+'/static')
CORS(app, support_credentials=True)

def generate_random_plot_id():
    return str(uuid.uuid4())

def save_plot_image(plot, plot_id):
    plot.save(f"plots/{plot_id}.png")
    

def delete_plot_image(plot_id):
    os.remove(f"plots/{plot_id}.png")

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
        try:
            uf = urllib.request.urlopen(link)
            html = uf.read()

            soup = BeautifulSoup(html, 'html.parser')

            page_text = soup.get_text()

            text_converter = html2text.HTML2Text()
            text += text_converter.handle(page_text)
        except HTTPError as e:
            if e.code == 403:
                print("403 Forbidden: You don't have permission to access this resource.")
            else:
                print(f"HTTP Error {e.code}: {e.reason}")

    #print(text)
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

    if len(text) > 4000:
        text = text[:4000]

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

def create_plot(exel_file, prompt):
    df = pd.read_excel(exel_file)

    plotai = PlotAI(df)
    plot_id = generate_random_plot_id()

    if not os.path.exists(ROOT_DIR + "/static/plots"):
        os.makedirs(ROOT_DIR + "/static/plots")

    plotai.make(prompt + ", do not show the plot, save the plot as "+ ROOT_DIR +"/static/plots/"+ plot_id +".png")

    return plot_id

chat_history = {}

@app.route('/docs', methods=['POST'])
@cross_origin(supports_credentials=True)
def docs():
    global chat_history
    if request.method == 'POST':
        if (development): return 'hello world'

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

        #print(messages)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)

        response_content = response.choices[0].message['content']

        # Almacena la conversación actual
        save_chat_history(chat_key, user_question)

        return response_content

@app.route('/plot', methods=['POST'])
@cross_origin(supports_credentials=True)
def plot():
    if request.method == 'POST':
        if (development): return ['adc3687f-e6f5-4c55-819c-44ad0cf8e54d','9230f682-144c-45ba-84a9-eb052ce5d751']
        #todo: haz que retorne directamente la imagen y en el frontent se cargue directamente el url en un src
        try:
            documents = request.files.getlist('documents[]')
            prompt = request.form['prompt']

            if documents and prompt:
                plots = []
                for document in documents:
                    excel_file = document.read()
                    plot = create_plot(excel_file, prompt)
                    plots.append(plot)

                return plots
            else:
                return ''
        except Exception as e:
            # Handle the exception here
            # For example, log the error and return a 500 Internal Server Error response
            print(e)
            return []
        
@app.route('/embebings', methods=['POST'])
@cross_origin(supports_credentials=True)
def documents():
    if request.method == 'POST':
        user_question = request.form['chat']

        text = get_text_from_request(request)
        chunks = split_text_into_chunks(text)

        knowledge_base = create_embeddings(chunks)

        if user_question:
            response = handle_user_question(knowledge_base, user_question)
            return response
        

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))