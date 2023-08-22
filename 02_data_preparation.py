# Databricks notebook source
# MAGIC %md Installing required python libraries

# COMMAND ----------

# MAGIC %pip install -U transformers==4.29.2 sentence-transformers==2.2.2 langchain==0.0.190 chromadb==0.3.25 pypdf==3.9.1 pycryptodome==3.18.0 accelerate==0.19.0 unstructured==0.7.1 unstructured[local-inference]==0.7.1 sacremoses==0.0.53 ninja==1.11.1 tiktoken
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import nltk
nltk.download('punkt')

# COMMAND ----------

# MAGIC %md Creating a dropdown widget for:
# MAGIC - model selection
# MAGIC - where our PDFs are stored
# MAGIC - where we want to cache the HuggingFace model downloads
# MAGIC - where we want to persist our vectorstore

# COMMAND ----------

# where you want the PDFs to be save in your environment
dbutils.widgets.text("PDF_Path", "/dbfs/tmp/hls_qa/pdfs")

# which embeddings model from Hugging Face 🤗 you would like to use; for biomedical applications we have been using this model recently
# also worth trying this model for embeddings for camparison: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
dbutils.widgets.text("Embeddings_Model", "pritamdeka/S-PubMedBert-MS-MARCO")

# where you want the vectorstore to be persisted across sessions, so that you don't have to regenerate
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/hls_qa/db")

# publily accessible bucket with PDFs for this demo
dbutils.widgets.text("Source_Documents", "s3a://db-gtm-industry-solutions/data/hls/llm_qa/")

# where you want the hugging face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

# get widget values
pdf_path = dbutils.widgets.get("PDF_Path")
source_pdfs = dbutils.widgets.get("Source_Documents")
db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")
embeddings_model = dbutils.widgets.get("Embeddings_Model")

# COMMAND ----------

import os
# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# COMMAND ----------

# MAGIC %md ## Data Prep
# MAGIC
# MAGIC This data preparation need only happen one time to create data sets that can then be reused in later sections without re-running this part.
# MAGIC
# MAGIC * Grab the set of PDFs (ex: Arxiv papers allow curl, PubMed does not)
# MAGIC * We have are providing a set of PDFs from PubMedCentral relating to Cystic Fibrosis (all from [PubMedCentral](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) Open Access, all with the CC BY license), but any topic area would work
# MAGIC * If you already have a repository of PDFs then you can skip this step, just organize them all in an accessible DBFS location

# COMMAND ----------

import os
import shutil

# COMMAND ----------

# in case you rerun this notebook, this deletes the directory and recreates it to prevent file duplication
if os.path.exists(pdf_path):
    shutil.rmtree(pdf_path, ignore_erros=True)
os.makedirs(pdf_path)

# COMMAND ----------

# slightly modify the file path from above to work with the dbuitls.fs sytax
modified_pdf_path = "dbfs:/" + pdf_path.lstrip("/dbfs")
dbutils.fs.cp(source_pdfs, modified_pdf_path, True)

# COMMAND ----------

# MAGIC %md
# MAGIC All of the PDFs should now be accessible in the `pdf_path` now; you can run the below command to check if you want.

# COMMAND ----------

!ls {pdf_path}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Document DB
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Use `langchain` to reading directly from PDFs, although LangChain also supports txt, HTML, Word docs, GDrive, PDFs, etc.
# MAGIC - Create a simple in-memory Chroma vector DB for storage
# MAGIC - Instantiate an embedding function from `sentence-transformers`
# MAGIC - Populate the database and save it

# COMMAND ----------

# MAGIC %md Prepare a directory to store the document database. Any path on `/dbfs` will do

# COMMAND ----------

!(rm -r {db_persist_path} || true) && mkdir - p {db_persist_path}

# COMMAND ----------

# MAGIC %md Create the document database:
# MAGIC * Here we are using the `PyPDFDirectoryLoader` loader from LangChain ([docs page](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pypdf)) to form `documents`; `langchain` can also from doc collections directory from PDFs, GDrive files etc.

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader

# COMMAND ----------

loader_path = f"{pdf_path}/"

# COMMAND ----------

pdf_loader = PyPDFDirectoryLoader(loader_path)

# COMMAND ----------

docs = pdf_loader.load()

# COMMAND ----------

len(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we are using a text splitter from LangChain to split our PDFs into manageable chunks. This is for a few reasons, primarily:
# MAGIC
# MAGIC - LLMs (currently) have a limited context length. MPT-7b-Instruct by default can only accept 2048 tokens (roughly words) in the prompt, although it can accept 4096 with a small settings change. This is rapidly changing, though, so keep an eye on it.
# MAGIC - When we create embeddings for these documents, an NLP model (sentence transformer) creates a numerical representation (a high-dimensional vector) of that chunk of text that captures the semantic meaning of what is being embedded. If we were to embed large documents, the NLP model would need to capture the meaning of the entire document in one vector; by splitting the document, we can capture the meaning of chunks throughout that document and retrieve only what is most relevant.
# MAGIC - In this case, the embeddings model we use can except a very limited number of tokens. The default one we have selected in this notebook, [S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO), has also been finetuned on a PubMed dataset, so it is particularly good at generating embeddings for medical documents.

# COMMAND ----------

# For PDFs we need to split them for embeddings:
from langchain.text_splitter import TokenTextSplitter

# COMMAND ----------

# this is plitting into chunks based on a fixed number of tokens
# the embeddings model we use below can take a maximum of 128 tokens (and truncates beyon that) so we keep our chunks at that max size
text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=32)

# COMMAND ----------

documents = text_splitter.split_documents(docs)

# COMMAND ----------

display(documents)

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# COMMAND ----------

hf_embed = HuggingFaceEmbeddings(model_name=embeddings_model)

# COMMAND ----------

sample_query = "What is cystic fibrosis?"

# COMMAND ----------

db = Chroma.from_documents(collection_name="hls_docs", documents=documents, embedding=hf_embed, persist_directory=db_persist_path)
db.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building a `langchain` Chain
# MAGIC Now we can compose the database with a language model and prompting strategy to make a `langchain` chain that answers questions.
# MAGIC
# MAGIC - Load the Chroma DB
# MAGIC - Instantiate an LLM, like Dolly here, but could be other models or even OpenAI models
# MAGIC - Define how relevant texts are combined with a question into the LLM prompt

# COMMAND ----------

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
 
db_persist_path = db_persist_path
hf_embed = HuggingFaceEmbeddings(model_name=embeddings_model)
db = Chroma(collection_name="hls_docs", embedding_function=hf_embed, persist_directory=db_persist_path)
