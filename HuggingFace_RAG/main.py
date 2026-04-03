import ollama

dataset = []

with open('cat-facts.txt', 'r', encoding='utf-8') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

# print(dataset)

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = [] # Using this as a database but ideally we should use chromadb or pinecone like that.

def add_chunk_to_db(chunk):
    embedding = ollama.embed(model= EMBEDDING_MODEL, input= chunk)["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):  # Here i am considering each line as a separate chunk
    add_chunk_to_db(chunk)
    print(f"Added {(i + 1) / len(dataset)} to the database")

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n = 3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key= lambda x: x[1], reverse=True)
    return similarities[:top_n]

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, _ in retrieved_knowledge])}
'''

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
