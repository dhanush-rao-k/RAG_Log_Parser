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

def view_logs(collection):
    """
    Display all logs stored in the ChromaDB collection.
    """

    data = collection.get()

    print("\nStored Logs:\n")

    for i in range(len(data["ids"])):

        print("=" * 60)
        print("ID:", data["ids"][i])

        print("\nMetadata:")
        print(data["metadatas"][i])

        print("\nDocument:")
        print(data["documents"][i])

    print("=" * 60)

def build_system_prompt():

    return """
You are an experienced SOC analyst.

You are given:

1. The CURRENT log.

2. Similar historical logs retrieved from a vector database.

Your job is to determine whether the current log is malicious.

Historical logs are context only.

Do not simply repeat their classification.

If the current log is normal, classify it as benign.

Return ONLY valid JSON.

{
    "severity":"",
    "attack_type":"",
    "confidence":0,
    "summary":"",
    "recommendation":""
}
"""
def build_user_prompt(current_log, retrieved_logs):

    prompt = f"""
Current Log

{current_log}

"""

    if len(retrieved_logs) > 0:

        prompt += "\nHistorical Similar Logs\n\n"

        for i, log in enumerate(retrieved_logs, start=1):

            prompt += f"Log {i}\n"

            prompt += log

            prompt += "\n\n"

    else:

        prompt += "\nNo similar historical logs found.\n\n"

    prompt += """
Instructions

1. Analyze ONLY the current log.

2. Historical logs are context only.

3. Determine whether this represents malicious activity.

4. If no anomaly exists, state that it is benign.

Return ONLY JSON.

"""

    return prompt

def classify_log(system_prompt, user_prompt):

    response = chat(

        model="mistral",

        messages=[

            {
                "role":"system",
                "content":system_prompt
            },

            {
                "role":"user",
                "content":user_prompt
            }

        ]

    )

    return response["message"]["content"]

def parse_output(response):

    response = response.replace("```json", "")

    response = response.replace("```", "")

    response = response.strip()

    return json.loads(response)

def print_output(result):

    print("\nSOC Analysis")

    print("-"*40)

    print("Severity       :", result["severity"])

    print("Attack Type    :", result["attack_type"])

    print("Confidence     :", result["confidence"])

    print("Summary        :", result["summary"])

    print("Recommendation :", result["recommendation"])

def analyze_without_rag(log):

    system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(log, [])

    response = classify_log(
        system_prompt,
        user_prompt
    )

    result = parse_output(response)

    print_output(result)

def analyze_with_rag(
        current_log,
        retrieved_logs
):

    print("\nRetrieved Context\n")

    for i, log in enumerate(retrieved_logs, start=1):

        print(f"Similar Log {i}")

        print(log)

        print("-"*50)

    system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(
        current_log,
        retrieved_logs
    )

    response = classify_log(
        system_prompt,
        user_prompt
    )

    result = parse_output(response)

    print_output(result)

def retrieve_context(
        collection,
        embedding,
        metadata,
        top_k=3
):

    where_clause = {}

    if metadata["src_ip"] != "unknown":
        where_clause["src_ip"] = metadata["src_ip"]

    if metadata["dst_ip"] != "unknown":
        where_clause["dst_ip"] = metadata["dst_ip"]

    if metadata["port"] != "unknown":
        where_clause["port"] = metadata["port"]

    try:

        if len(where_clause) > 0:

            results = collection.query(

                query_embeddings=[embedding],

                n_results=top_k + 1,

                where=where_clause

            )

        else:

            results = collection.query(

                query_embeddings=[embedding],

                n_results=top_k + 1

            )

    except:

        results = collection.query(

            query_embeddings=[embedding],

            n_results=top_k + 1

        )

    retrieved_logs = []

    documents = results["documents"][0]

    distances = results["distances"][0]

    for doc, distance in zip(documents, distances):

        if distance < 0.00001:
            continue

        retrieved_logs.append(doc)

        if len(retrieved_logs) == top_k:
            break

    return retrieved_logs



def main():

    collection = initialize_db()

    print("=" * 60)
    print("SOC RAG Log Classifier Started")
    print("=" * 60)

    while True:

        # Take log input and store it
        log, embedding, metadata = process_log(collection)

        count = get_log_count(collection)

        print(f"\nCurrent Logs in Database: {count}")

        # Run without RAG until enough logs are available
        if count < 5:

            print("\nLess than 5 logs available.")
            print("Running normal LLM classification...\n")

            analyze_without_rag(log)

        else:

            print("\n5 or more logs detected.")
            print("Running RAG-enhanced classification...\n")

            # Will be implemented in the next step
            retrieved_logs = retrieve_context(
                collection,
                embedding,
                metadata
            )

            analyze_with_rag(
                log,
                retrieved_logs
            )

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()