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
    "Como o livro Algoritmos: Teoria e Prática, de Cormen, define a notação Θ (Theta) e qual teorema relaciona Θ com as notações O e Ω?",
    "Como Manzano e Oliveira, no livro Algoritmos: Lógica para Desenvolvimento de Programação de Computadores, descrevem o papel do programador de computador e o que é o diagrama de blocos?",
    "Segundo Dilermando Junior e Nakamiti em Algoritmos e Programação de Computadores, qual é a origem do termo \"algoritmo\" e em que consiste o Algoritmo Euclidiano para o cálculo do mdc?",
    "Por que, segundo Sebesta no livro Conceitos de Linguagens de Programação, é importante estudar os conceitos de linguagens de programação mesmo para quem não vai criar uma nova linguagem?",
    "Como Bhargava, no livro Entendendo Algoritmos, define a notação Big O e o que ela estabelece sobre o tempo de execução de um algoritmo?",
    "Segundo Szwarcfiter em Estruturas de Dados e Seus Algoritmos, quais são as complexidades das operações de seleção, inserção, remoção, alteração e construção em um heap?",
    "Como Ascencio, no livro Fundamentos da Programação de Computadores, descreve a plataforma Java, os arquivos gerados na compilação e o papel da Máquina Virtual Java?",
    "Segundo o livro Introdução a Algoritmos e Programação, quais são as três partes que compõem um algoritmo executado em um computador e quais sistemas de representação numérica são utilizados internamente?",
    "Quais são as quatro perguntas que Nilo Menezes, em Introdução à Programação com Python, recomenda que o iniciante responda antes de começar a aprender a programar e qual é, segundo o autor, a maneira mais difícil de aprender?",
    "Quais são os operadores aritméticos não convencionais apresentados por Forbellone em Lógica de Programação e como o autor define o conceito de contador?"
]

ground_truths = [
    "Cormen define que, para uma função g(n), Θ(g(n)) representa o conjunto de funções com limites assintóticos justos: existe um limite superior e inferior do mesmo crescimento. O Teorema 3.1 do livro estabelece que, para quaisquer duas funções f(n) e g(n), tem-se f(n) = Θ(g(n)) se e somente se f(n) = O(g(n)) e f(n) = Ω(g(n)). Em outras palavras, uma função tem ordem Θ exatamente quando possui simultaneamente o mesmo limite assintótico superior (O) e inferior (Ω).",
    "Manzano e Oliveira comparam o programador a um construtor (ou pedreiro especializado), responsável por construir o programa empilhando instruções de uma linguagem como se fossem tijolos, inclusive elaborando a interface gráfica. Além de interpretar o fluxograma desenhado pelo analista, o programador deve detalhar a lógica do programa em nível micro, desenhando uma planta operacional chamada diagrama de blocos (ou diagrama de quadros), seguindo a norma ISO 5807:1985. Essa atividade exige alto grau de atenção e cuidado, pois o descuido pode \"matar\" uma empresa.",
    "Segundo Dilermando Junior e Nakamiti, o termo \"algoritmo\" deriva do nome do matemático persa al-Khwarizmi, considerado por muitos o \"Pai da Álgebra\". No século XII, Adelardo de Bath traduziu uma de suas obras para o latim, registrando o termo como \"Algorithmi\"; originalmente referia-se às regras de aritmética com algarismos indo-arábicos e, posteriormente, passou a designar qualquer procedimento definido para resolver problemas. O Algoritmo Euclidiano, criado por Euclides, calcula o máximo divisor comum (mdc): divide-se a por b, obtendo o resto r; substitui-se a por b e b por r; e repete-se a divisão até que não seja mais possível dividir, sendo o último valor de a o mdc.",
    "Sebesta argumenta que estudar conceitos de linguagens valoriza recursos e construções importantes e estimula o programador a usá-los mesmo quando a linguagem em uso não os suporta diretamente — por exemplo, simulando matrizes associativas de Perl em outra linguagem. Também fornece embasamento para escolher a linguagem mais adequada a cada projeto, evitando que o profissional se restrinja àquela com a qual está mais familiarizado. Por fim, conhecer uma gama mais ampla de linguagens torna o aprendizado de novas linguagens mais fácil, ampliando a capacidade de avaliar trade-offs de projeto.",
    "Bhargava define a notação Big O como uma forma de medir o tempo de execução de um algoritmo no pior caso (pior hipótese), descrevendo o quão rapidamente esse tempo cresce em relação ao tamanho n da entrada. Por exemplo, a pesquisa simples tem tempo O(n) — no pior caso verifica todos os elementos da lista — enquanto a pesquisa binária tem tempo O(log n). Algoritmos com tempos diferentes crescem a taxas muito distintas, e o Big O permite compará-los independentemente do hardware utilizado.",
    "Segundo Szwarcfiter, em um heap o elemento de maior prioridade é sempre a raiz da árvore, e as operações têm os seguintes parâmetros de eficiência: seleção em O(1), pois basta retornar a raiz; inserção em O(log n); remoção em O(log n); alteração em O(log n); e construção em O(n), tempo este inferior ao de uma ordenação. Esses tempos tornam o heap especialmente adequado para implementar listas de prioridades.",
    "Ascencio explica que a tecnologia Java é composta pela linguagem de programação Java e pela plataforma de desenvolvimento Java, com características de simplicidade, orientação a objetos, portabilidade, alta performance e segurança. Os programas são escritos em arquivos de texto com extensão .java e, ao serem compilados pelo compilador javac, geram arquivos .class compostos por bytecodes — código interpretado pela Máquina Virtual Java (JVM). A plataforma Java é composta apenas por software, pois é a JVM que faz a interface entre os programas e o sistema operacional.",
    "O livro descreve que um algoritmo, quando programado em um computador, é constituído por pelo menos três partes: entrada de dados, processamento de dados e saída de dados. Internamente, os computadores digitais utilizam o sistema binário (base 2), com apenas dois algarismos (0 e 1), aproveitando a noção de ligado/desligado ou verdadeiro/falso. Como representações auxiliares, são também utilizados o sistema decimal (base 10), o sistema hexadecimal (base 16, com dígitos 0–9 e A–F) e o sistema octal (base 8).",
    "Menezes propõe que o iniciante responda a quatro perguntas antes de começar: (1) Você quer aprender a programar?; (2) Como está seu nível de paciência?; (3) Quanto tempo você pretende estudar?; (4) Qual o seu objetivo ao programar? Para o autor, a maneira mais difícil de aprender a programar é não querer programar — a vontade deve vir do próprio aluno e não de um professor ou amigo. Programar é uma arte que exige tempo, dedicação e paciência para que a mente se acostume com a nova forma de pensar.",
    "Forbellone apresenta operadores aritméticos não convencionais úteis na construção de algoritmos: pot(x,y) para potenciação (x elevado a y), rad(x) para radiciação (raiz quadrada de x), mod para o resto da divisão (ex.: 9 mod 4 = 1) e div para o quociente da divisão inteira (ex.: 9 div 4 = 2). Um contador é uma variável usada para registrar quantas vezes um trecho de algoritmo é executado: é declarada com um valor inicial e incrementada (somada de uma constante, normalmente 1) a cada repetição, comportando-se como o ponteiro dos segundos de um relógio."
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
