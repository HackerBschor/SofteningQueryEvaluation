# Softening Structured Query Answering with Large Language Models 🦙🖥️🎓

## Model Download
In this project, we work with LLama by Meta. 
So, we start by downloading the model from the [official download page](https://www.llama.com/llama-downloads/). 
To use the model with the Hugging Face classes, we need to convert the model using a 
[transform script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py).

We've used [`Llama3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) for text generation 
and [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) for text embedding. 

```bash
python convert_llama_weights_to_hf.py --model_size <size> --llama_version <version> --input_dir <model> --output_dir <model>_compile
# python convert_llama_weights_to_hf.py --model_size 3B --llama_version 3.2 --input_dir Llama3.2-3B  --output_dir Llama3.2-3B_compile 
```

## Database Configuration

As backbone of this project, we chose [PostgreSQL](https://www.postgresql.org/) together with the 
[pgvector](https://github.com/pgvector/pgvector) extension, so one has to install both as stated on their Website and GitHub.

Then we have to create a database 
```sql
CREATE DATABASE <db>;
CREATE USER <user> WITH ENCRYPTED PASSWORD '<password>';
ALTER DATABASE <db> OWNER TO <user>;
GRANT ALL PRIVILEGES ON DATABASE <db> TO <user>;
```

Then, activate the [pgvector](https://github.com/pgvector/pgvector) extension and create the schema for the embeddings

```sql
/* String Similarity */
CREATE EXTENSION pg_trgm;

/* Activate pgvector */
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

/* create schema */
CREATE SCHEMA embeddings;
```



## Configuration

```ini
[DB]
database=<database>
host=<host>
user=<user>
password=<password>
port=<port>

[MODEL]
path_generation=<path_to_transformed_llama_model>
path_embeddings=intfloat/e5-base-v2
```