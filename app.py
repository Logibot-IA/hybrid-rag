import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

load_dotenv()

test_queries = [
    "O que é lógica proposicional segundo a apostila?",
    "Como a apostila define uma proposição?",
    "O que são conectivos lógicos e quais são apresentados no material?",
    "O que é uma tabela-verdade e para que ela é utilizada?",
    "Como a apostila define tautologia, contradição e contingência?"
]

ground_truths = [
    "Lógica proposicional é o ramo da lógica que estuda proposições e as relações entre elas por meio de conectivos lógicos.",
    "Proposição é toda sentença declarativa que pode ser classificada como verdadeira ou falsa, mas não ambas.",
    "Conectivos lógicos são operadores que conectam proposições, como negação (¬), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "Tabela-verdade é um método utilizado para determinar o valor lógico de proposições compostas a partir dos valores lógicos das proposições simples.",
    "Tautologia é uma proposição composta que é sempre verdadeira; contradição é sempre falsa; contingência é aquela que pode ser verdadeira ou falsa dependendo dos valores das proposições componentes."
]

# 1. Setup
model = ChatOpenAI(
    model="llama3.3-70b-instruct",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://inference.do-ai.run/v1",
    temperature=0.7,
    max_tokens=1024,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2. Indexação — Carregar e dividir documentos
loader = PyPDFDirectoryLoader("../docs/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=120,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
print(f"Split doc into {len(all_splits)} sub-documents.")

# 3. Retriever Vetorial (Dense / Semântico)
vector_store = Chroma(
    collection_name="hybrid_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
vector_store.add_documents(documents=all_splits)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. Retriever BM25 (Sparse / Lexical)
bm25_retriever = BM25Retriever.from_documents(all_splits, k=5)

# 5. Hybrid Retriever (Ensemble com RRF)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

# 6. RAG Chain (Prompt + LLM)
prompt = ChatPromptTemplate.from_template(
    """Você é um assistente útil. Use o contexto abaixo para responder a pergunta.
Se não souber a resposta com base no contexto, diga que não sabe.

Contexto:
{context}

Pergunta: {question}

Resposta:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


def evaluate_with_ragas():
    eval_llm = ChatOpenAI(
        model="llama3.3-70b-instruct",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://inference.do-ai.run/v1",
        temperature=0,
    )

    print("Coletando respostas para avaliacao RAGAS...\n")
    ragas_data = []

    for i, query in enumerate(test_queries):
        print(f"  [{i+1}/{len(test_queries)}] {query}")
        answer = rag_chain.invoke(query)
        context_docs = hybrid_retriever.invoke(query)
        contexts = [doc.page_content for doc in context_docs]

        ragas_data.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truths[i]
        })

    dataset = Dataset.from_list(ragas_data)

    print("\nExecutando avaliacao RAGAS...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=embeddings,
    )

    print("\n=== RESULTADOS RAGAS ===")
    print(result)

    df = result.to_pandas()
    print("\nDetalhes por query:")
    print(df.to_string())

    return result


if __name__ == "__main__":
    evaluate_with_ragas()
