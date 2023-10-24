import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores import PGEmbedding

from langchain.chains import NebulaGraphQAChain
from langchain.graphs import NebulaGraph

from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

import psycopg2


from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    Vector_DataBase
)

def file_log(logentry):
   file1 = open("file_ingest.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
       file_extension = os.path.splitext(file_path)[1]
       loader_class = DOCUMENT_MAP.get(file_extension)
       if loader_class:
           file_log(file_path + ' loaded.')
           loader = loader_class(file_path)
       else:
           file_log(file_path + ' document type is undefined.')
           raise ValueError("Document type is undefined")
       return loader.load()[0]
    except Exception as ex:
       file_log('%s loading error: \n%s' % (file_path, ex))
       return None 

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
           file_log(name + ' failed to submit')
           return None
        else:
           data_list = [future.result() for future in futures]
           # return data and file paths
           return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print('Importing: ' + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
               future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
               file_log('executor task failed: %s' % (ex))
               future = None
            if future is not None:
               futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log('Exception: %s' % (ex))
                
    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == ".py":
               python_docs.append(doc)
           else:
               text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200)
    
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")
    
    print(texts[0])

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )
    # change the embedding type here if you are running into issues.
    # These are much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if Vector_DataBase == 'Chroma_DB':
        # Chroma Database
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )
        
    elif Vector_DataBase == 'FAISS_DB':
        # Faiss Database
        db = FAISS.from_documents(texts, 
                                embeddings
                                )
        
        db.save_local(PERSIST_DIRECTORY, "faiss_index")
    
    elif Vector_DataBase == 'PGvector_DB':
        # PGVector Database
        COLLECTION_NAME = "Saudi_Sanitaryware_Market"
        
        host= os.environ['m5ep5eubs6.yjhj90cwyu.tsdb.cloud.timescale.com']
        port= os.environ['38257']
        user= os.environ['tsdbadmin']
        password= os.environ['gcijpmi7hele9p25']
        dbname= os.environ['tsdb']

        # CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        
        
        #conn = psycopg2.connect(CONNECTION_STRING)
        #cursor = conn.cursor()

        db = PGVector.from_documents(
            embedding=embeddings,
            documents=texts,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING)
    
    elif Vector_DataBase == 'PG_Embeddings':

        # PG Embeddings 
        collection_name = "Saudi_Sanitaryware_Market"
        connection_string = ''
        
        db = PGEmbedding.from_documents(
            embedding=embeddings,
            documents=texts,
            collection_name=collection_name,
            connection_string=connection_string,)

    elif Vector_DataBase == 'Neo4j_DB':
        
        NEO4J_URI = 'neo4j+s://0efbb97d.databases.neo4j.io'
        NEO4J_USERNAME = 'neo4j'
        NEO4J_PASSWORD = 'W2UEItdSdfwfu-Ej-Fa7jwdynACScY7LKaY4hw_nW0w'
        AURA_INSTANCEID = '0efbb97d'
        AURA_INSTANCENAME = 'Instance01'
        
        graph = Neo4jGraph(
                url=NEO4J_URI, 
                username=NEO4J_USERNAME, 
                password=NEO4J_PASSWORD)
        
        graph.query(texts)
        
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0), graph=graph, verbose=True)
    

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
