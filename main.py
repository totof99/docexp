from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from itertools import batched
import uuid


def print_progress_bar(current, total, length=100):
    percent = 100*(current/float(total))
    filled = int(length * current // total)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\rTreated chunks: |{bar}| {current: >{len(str(total))}}/{total} {percent: >3.0f}%', end='')
    if current == total: 
        print()


filePath = "documents/basel_minimum_capital_requirements_for_market_risk.pdf"
ollama_url="http://localhost:11434"

print("Loading " + filePath)
loader = PyPDFLoader(filePath)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = loader.load_and_split(text_splitter=text_splitter)
print("Loaded " + str(len(documents)) + " chunks from PDF " + filePath)

ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]

unique_documents_ids = {}
for doc, id in zip(documents, ids):
    if id not in unique_documents_ids:
        unique_documents_ids[id] = doc

print("Found " + str(len(unique_documents_ids)) + " unique chunks")

oembed = OllamaEmbeddings(base_url=ollama_url, model="mistral")
vectorstore = Chroma(embedding_function=oembed, persist_directory="chroma")

total_chunks = len(unique_documents_ids)
chunks_treated = 0

print_progress_bar(chunks_treated, total_chunks)
for docs_with_ids in batched(unique_documents_ids.items(), 10):
    nb_docs = len(docs_with_ids)
    ids = [id for id, _ in docs_with_ids]
    docs = [doc for _, doc in docs_with_ids]
   
    results = vectorstore.get(ids=ids, include=[])
    if len(results['ids']) > 0:
        existing_ids = set(results['ids'])
        ids = [id for id, _ in docs_with_ids if id not in existing_ids]
        docs = [doc for id, doc in docs_with_ids if id not in existing_ids]

    if len(docs) > 0:
        vectorstore.add_documents(documents=docs, ids=ids)
    
    chunks_treated += nb_docs
    print_progress_bar(chunks_treated, total_chunks)

print("Vector database filled with embeddings")

ollama = Ollama(base_url=ollama_url, model="mistral")
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
answer=qachain.invoke({"query": "how do you compute SA for equity asset class?"})
print(answer)