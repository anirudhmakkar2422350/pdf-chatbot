from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================================
# 1. LOAD THE PDF
# ================================
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print("PDF loaded successfully!")
print(f"Total pages loaded: {len(documents)}")

# ================================
# 2. SPLIT TEXT INTO CHUNKS
# ================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# ================================
# 3. CREATE EMBEDDINGS (FREE)
# ================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings object created successfully!")

# ================================
# 4. CREATE VECTOR STORE (FAISS)
# ================================
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully!")

# ================================
# 5. INTERACTIVE QUESTION ANSWERING
# ================================
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting PDF Chatbot 👋")
        break

    docs = vectorstore.similarity_search(query, k=2)

    print("\nRetrieved chunks:")
    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}:\n{doc.page_content}")

    # ================================
    # 6. SIMPLE ANSWER GENERATION
    # ================================
    context = "\n\n".join(doc.page_content for doc in docs)

    print("\nFINAL ANSWER:")
    print("---------------------------------")
    print(context)
    print("---------------------------------")
