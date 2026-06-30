import json
import uuid
import re

import chromadb
from ollama import chat, embeddings

def initialize_db():
    """
    Initialize ChromaDB and create/get the logs collection taht will store the logs.
    """

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="logs"
    )

    return collection

def generate_embedding(log):
    """The log is passed and the embedding of the log is returned """

    response = embeddings(
        model="nomic-embed-text",
        prompt=log
    )

    return response["embedding"]

def extract_metadata(log):
    """Regex based extractor to extract src_ip, dst_ip, port and timestamp from the log. If not found, it will return 'unknown' for that field."""
    metadata = {}

    src_ip = re.search(r"src(?:_ip)?[:= ]+(\d+\.\d+\.\d+\.\d+)", log, re.I)
    dst_ip = re.search(r"dst(?:_ip)?[:= ]+(\d+\.\d+\.\d+\.\d+)", log, re.I)

    port = re.search(r"port[:= ]+(\d+)", log, re.I)

    timestamp = re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        log
    )

    metadata["src_ip"] = src_ip.group(1) if src_ip else "unknown"

    metadata["dst_ip"] = dst_ip.group(1) if dst_ip else "unknown"

    metadata["port"] = port.group(1) if port else "unknown"

    metadata["timestamp"] = (
        timestamp.group(0)
        if timestamp
        else "unknown"
    )

    return metadata

def store_log(
        collection,
        log,
        embedding,
        metadata
):

    """Store the log, embedding and metadata in the Vector DB."""

    log_id = str(uuid.uuid4())

    collection.add(

        ids=[log_id],

        embeddings=[embedding],

        documents=[log],

        metadatas=[metadata]

    )

def get_log_count(collection):

    return collection.count()

def get_log():
    """Get the log from the user """

    print()

    print("Paste the log.")

    print("Type END on a new line.")

    lines = []

    while True:

        line = input()

        if line.upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines)

def process_log(collection):

    log = get_log()

    embedding = generate_embedding(log)

    metadata = extract_metadata(log)

    store_log(
        collection,
        log,
        embedding,
        metadata
    )

    print()

    print("Log stored successfully.")

    print(metadata)

    return log, embedding, metadata

def main():

    collection = initialize_db()

    while True:

        process_log(collection)

        print()

        print(
            "Current logs:",
            get_log_count(collection)
        )

        print(view_logs(collection))

def view_logs(collection):

    data = collection.get()

    print("\nStored Logs:\n")

    for i in range(len(data["ids"])):

        print("=" * 50)
        print("ID:", data["ids"][i])
        print("\nMetadata:")
        print(data["metadatas"][i])
        print("\nDocument:")
        print(data["documents"][i])

if __name__ == "__main__":
    main()

