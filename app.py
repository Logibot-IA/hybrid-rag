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

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "benchmark-hybrid-rag")

test_queries = [
    "O que são conectivos lógicos e quais são os cinco conectivos apresentados na Apostila de Lógica de Programação?",
    "Como a Apostila de Lógica de Programação define tabela-verdade e qual princípio a fundamenta?",
    "Como funciona a pesquisa binária e qual é a sua complexidade de tempo segundo o livro Entendendo Algoritmos?",
    "Qual é a estratégia de dividir para conquistar utilizada pelo quicksort conforme descrito em Entendendo Algoritmos?",
    "Quais são os principais domínios de programação e suas linguagens associadas segundo Sebesta em Conceitos de Linguagens de Programação?",
    "Como Szwarcfiter define complexidade de pior caso de um algoritmo no livro Estruturas de Dados e Seus Algoritmos?",
    "Como o livro Fundamentos da Programação de Computadores de Ascencio descreve a função do computador e a necessidade de algoritmos?",
    "Como Forbellone define algoritmo e o conceito de sequenciação no livro Lógica de Programação?",
    "Segundo Nilo Menezes no livro Introdução à Programação com Python, o que realmente significa saber programar?",
    "Como Manzano define o termo algoritmo no livro Algoritmos: Lógica para Desenvolvimento de Programação de Computadores e qual é a sua origem etimológica?"
]

ground_truths = [
    "Conectivos lógicos são palavras ou frases usadas para formar novas proposições a partir de outras proposições. Os cinco conectivos apresentados são: negação (~), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "A tabela-verdade é um dispositivo que apresenta todos os possíveis valores lógicos de uma proposição composta, correspondentes a todas as atribuições de valores às proposições simples. Ela é fundamentada no Princípio do Terceiro Excluído, segundo o qual toda proposição é verdadeira ou falsa, e o valor lógico de uma proposição composta depende unicamente dos valores lógicos das proposições simples componentes.",
    "A pesquisa binária funciona eliminando metade dos elementos a cada iteração, comparando o elemento central com o valor buscado. Para uma lista de n elementos, requer no máximo log₂n verificações, resultando em complexidade O(log n), muito mais eficiente que a pesquisa simples O(n) para listas grandes.",
    "A estratégia de dividir para conquistar consiste em identificar o caso-base mais simples e dividir o problema até chegar a ele. O quicksort escolhe um elemento pivô, separa o array em dois subarrays com elementos menores e maiores que o pivô, e ordena recursivamente cada subarray. No caso médio, possui complexidade O(n log n).",
    "Sebesta identifica cinco domínios principais: aplicações científicas (Fortran, ALGOL), aplicações comerciais (COBOL), inteligência artificial (LISP, Prolog), programação de sistemas (C, C++) e programação para a Web. Cada domínio possui características distintas que motivaram o desenvolvimento de linguagens específicas.",
    "A complexidade de pior caso é definida como o número máximo de passos que um algoritmo executa considerando todas as entradas possíveis: max Ei ∈ E {ti}. Ela fornece um limite superior para o tempo de execução em qualquer situação e é a medida mais utilizada por garantir que o algoritmo nunca ultrapassará esse limite independentemente da entrada.",
    "Segundo Ascencio, o computador é uma máquina que recebe, manipula e armazena dados, mas não possui iniciativa, independência ou criatividade, precisando de instruções detalhadas. Sua finalidade principal é o processamento de dados: receber dados de entrada, realizar operações e gerar uma resposta de saída, o que requer algoritmos bem definidos.",
    "Forbellone define algoritmo como um conjunto de regras formais para a obtenção de um resultado ou da solução de um problema. A sequenciação é uma convenção que rege o fluxo de execução do algoritmo, determinando a ordem das ações de forma linear, de cima para baixo, assim como se lê um texto, estabelecendo um padrão de comportamento seguido por qualquer pessoa.",
    "Segundo Menezes, saber programar não é decorar comandos, parâmetros e nomes estranhos. Programar é saber utilizar uma linguagem de programação para resolver problemas, ou seja, saber expressar uma solução por meio de uma linguagem de programação. A sintaxe pode ser esquecida, mas quem realmente sabe programar tem pouca dificuldade ao aprender uma nova linguagem.",
    "Segundo Manzano, o termo algoritmo deriva do nome do matemático Muhammad ibn Musā al-Khwārizmī. Do ponto de vista computacional, algoritmo é entendido como regras formais, sequenciais e bem definidas a partir do entendimento lógico de um problema, com o objetivo de transformá-lo em um programa executável por um computador, onde dados de entrada são transformados em dados de saída."
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
