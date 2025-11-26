from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === CONFIGURATION ===
print("=" * 60)
print("🤖 AGENT RAG - Questions/Réponses sur les CGU")
print("=" * 60)

# Charger les embeddings (même modèle que pour l'indexation)
print("\n⏳ Chargement du modèle d'embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Charger la base vectorielle existante
print("⏳ Connexion à la base vectorielle...")
persist_directory = "./chroma_db_cgu"
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="cgu_collection"
)
print(f"✅ Base chargée: {vectordb._collection.count()} documents")

# Configurer le retriever (récupère les k chunks les plus pertinents)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Charger le LLM local (Ollama avec Mistral)
print("⏳ Connexion au LLM local (Ollama/Mistral)...")
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
print("✅ LLM connecté")

# Prompt personnalisé pour le RAG
prompt_template = """Tu es un assistant juridique. Réponds de manière concise et directe (2-3 phrases maximum).
Utilise uniquement le contexte fourni.

Contexte:
{context}

Question: {question}

Réponse courte:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Fonction pour formater les documents en texte
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Créer la chaîne RAG avec LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# === FONCTION DE QUESTION/RÉPONSE ===
def ask_question(question: str):
    print(f"\n❓ {question}")
    
    # Exécuter la requête RAG
    response = rag_chain.invoke(question)
    
    # Afficher la réponse
    print(f"📝 {response}\n")
    
    return response

# === TESTS ===
print("\n" + "=" * 60)
print("🧪 5 QUESTIONS SUR LES CGU")
print("=" * 60)

# 5 questions simples sur le document CGU
questions = [
    "Qui est l'éditeur du site ?",
    "Que se passe-t-il si je divulgue mon mot de passe ?",
    "Est-ce que je peux utiliser le contenu du site à des fins commerciales ?",
    "Quel tribunal est compétent en cas de litige ?",
    "Quels sont mes droits sur mes données personnelles ?"
]

for question in questions:
    ask_question(question)

print("=" * 60)
print("✅ Test terminé !")
print("=" * 60)
