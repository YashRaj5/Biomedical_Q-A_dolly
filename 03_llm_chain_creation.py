# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC We are installing some NVIDIA libraries here in order to use a special package called Flash Attention (`flash_attn`) that Mosaic ML's mpt-7b-instruct model needs. This can take 5-10 minutes

# COMMAND ----------

# MAGIC %pip install -U transformers==4.29.2 sentence-transformers==2.2.2 langchain==0.0.190 chromadb==0.3.25 pypdf==3.9.1 pycryptodome==3.18.0 accelerate==0.19.0 unstructured==0.7.1 unstructured[local-inference]==0.7.1 sacremoses==0.0.53 ninja==1.11.1 torch==2.0.1 pytorch-lightning==2.0.1 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC only run this step if you are running below Databricks runtime 13.2 GPU; if using 13.2 and above this is not needed.

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# MAGIC sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# MAGIC sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# MAGIC sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
# MAGIC sudo apt-get update

# COMMAND ----------

# MAGIC %sh
# MAGIC apt-get install -y libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7

# COMMAND ----------

# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

import nltk

# COMMAND ----------

# which LLM do you want to use? You can grab LLM names from Hugging Face and replace/add them here if you want
dbutils.widgets.dropdown('model_name','mosaicml/mpt-7b-instruct',['databricks/dolly-v2-7b','tiiuae/falcon-7b-instruct','mosaicml/mpt-7b-instruct'])

# which embeddings model from Hugging Face 🤗 you would like to use; for biomedical applications we have been using this model recently
# also worth trying this model for embeddings for camparison: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
dbutils.widgets.text("Embeddings_Model", "pritamdeka/S-PubMedBert-MS-MARCO")

# where you want the vectorstore to be persisted across sessions, so that you don't have to regenerate
dbutils.widgets.text("Vectorstore_Persist_Path", "/dbfs/tmp/hls_qa/db")

# where you want the hugging face models to be temporarily saved
hf_cache_path = "/dbfs/tmp/cache/hf"

# COMMAND ----------

#get widget values
db_persist_path = dbutils.widgets.get("Vectorstore_Persist_Path")
embeddings_model = dbutils.widgets.get("Embeddings_Model")

# COMMAND ----------

import os
# Optional, but helpful to avoid re-downloading the weights repeatedly. Set to any `/dbfs` path.
os.environ['TRANSFORMERS_CACHE'] = hf_cache_path
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# COMMAND ----------


