from os import environ, path
from typing import List
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging  # Import logging
from termcolor import cprint
from transformers import AutoTokenizer

# CONSTANTS =====================================================
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "mixtral-8x7b-32768"
LLM_TEMPERATURE = 0.1

# Maximum chunk size allowed by the chosen embedding model
CHUNK_SIZE = 8192

CSV_FILE_PATH = "./file/data.csv"  # Path to the CSV file
VECTOR_STORE_DIR = "./vectorstore/"  # Directory where vectors are stored
COLLECTION_NAME = "collection1"  # chromadb collection name
# ===============================================================

# Configure logging
logging.basicConfig(
    filename='application.log',  # Name of the log file
    level=logging.INFO,           # Set the log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
)

load_dotenv()

def load_csv_data() -> List[Document]:
    """Loads the CSV file and converts it to a list of Documents."""
    try:
        logging.info("[+] Loading CSV data...")

        # Read the CSV file
        df = pd.read_csv(CSV_FILE_PATH)
        logging.info(f"[+] CSV data loaded, total rows: {len(df)}")

        # Convert each row to a Document object
        documents = []
        for idx, row in df.iterrows():
            content = " ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(page_content=content))

        return documents
    except Exception as e:
        logging.error(f"[-] Error loading the CSV file: {e}")


def chunk_document(documents: List[Document]) -> List[Document]:
    """Splits the input documents into maximum of CHUNK_SIZE chunks."""
    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/" + EMBED_MODEL_NAME, cache_dir=environ.get("HF_HOME")
    )
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE // 50,
    )

    logging.info(f"[+] Splitting documents...")
    chunks = text_splitter.split_documents(documents)
    logging.info(f"[+] Document splitting done, {len(chunks)} chunks total.")

    return chunks


def create_and_store_embeddings(
        embedding_model: JinaEmbeddings, chunks: List[Document]
) -> Chroma:
    """Calculates the embeddings and stores them in a Chroma vectorstore."""
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )

    # Process documents in smaller batches
    batch_size = 166  # Chroma's limit on batch size
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i: i + batch_size]
        texts = [doc.page_content for doc in batch_chunks]
        metadatas = [doc.metadata for doc in batch_chunks]

        vectorstore._collection.upsert(
            texts=texts,
            metadatas=metadatas,
            ids=[str(i) for i in range(i, i + len(batch_chunks))]
        )

    logging.info("[+] Vectorstore created with batched embeddings.")
    return vectorstore


def get_vectorstore_retriever(embedding_model: JinaEmbeddings) -> VectorStoreRetriever:
    """Returns the vectorstore."""
    db = chromadb.PersistentClient(VECTOR_STORE_DIR)
    try:
        # Check for the existence of the vectorstore specified by the COLLECTION_NAME
        db.get_collection(COLLECTION_NAME)
        retriever = Chroma(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        ).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logging.warning(f"[-] Collection not found: {e}. Creating a new one.")
        # The vectorstore doesn't exist, so create it.
        csv_data = load_csv_data()
        chunks = chunk_document(csv_data)
        retriever = create_and_store_embeddings(embedding_model, chunks).as_retriever(
            search_kwargs={"k": 3}
        )

    return retriever


def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq) -> Runnable:
    """Creates the RAG chain for course recommendations based on user input."""
    template = """Based on the user's learning preferences, suggest appropriate courses from the CSV file. 
    The user might describe their level of expertise or specific subjects they are interested in.

    <context>
    {context}
    </context>

    User Input: {input}

    Please suggest suitable courses for the user, or inform them if no relevant data is available.
    If no courses match the input, respond with: "Data not available for this."
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = get_vectorstore_retriever(embedding_model)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain



def run_chain(chain: Runnable) -> None:
    """Run the RAG chain with the user query."""
    while True:
        query = input("Enter a prompt: ")
        if query.lower() in ["q", "quit", "exit"]:
            logging.info("[+] User exited the prompt.")
            return
        logging.info(f"[+] Running the RAG chain with query: {query}")
        response = chain.invoke({"input": query})

        for doc in response["context"]:
            logging.info(f"[+] {doc.metadata} | content: {doc.page_content[:20]}...")

        logging.info(f"[+] Response: {response['answer']}")
        cprint("\n" + response["answer"], end="\n\n", color="light_blue")


def main() -> None:
    embedding_model = JinaEmbeddings(
        jina_api_key=environ.get("JINA_API_KEY"),
        model_name=EMBED_MODEL_NAME,
    )

    llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=LLM_NAME)

    chain = create_rag_chain(embedding_model=embedding_model, llm=llm)

    run_chain(chain)


if __name__ == "__main__":
    main()
