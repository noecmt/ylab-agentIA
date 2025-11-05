from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage



# Initialiser le modèle Ollama
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")

# Initialiser le store de vecteurs avec HuggingFace Embeddings (plus fiable)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Stocker l'historique de conversation (avec résumé automatique)
conversation_messages = []
MAX_HISTORY_LENGTH = 10  # Garder seulement les 10 derniers messages

# Template de prompt avec résumé
template = """Tu es un assistant IA serviable et amical. Utilise l'historique de conversation pour répondre de manière cohérente.

{history}

Humain: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Créer la chaîne avec LangChain
chain = prompt | llm | StrOutputParser()

def summarize_old_messages():
    """
    Résume les anciens messages si l'historique devient trop long.
    """
    global conversation_messages
    if len(conversation_messages) > MAX_HISTORY_LENGTH:
        # Garder les 3 premiers messages et résumer les autres
        old_messages = conversation_messages[3:-3]
        recent_messages = conversation_messages[-3:]
        
        # Créer un résumé des anciens messages
        summary_prompt = f"Résume brièvement cette conversation en 2-3 phrases:\n"
        for msg in old_messages:
            summary_prompt += f"{msg}\n"
        
        summary = llm.invoke(summary_prompt)
        
        # Remplacer par le résumé
        conversation_messages = [
            f"[Résumé des échanges précédents: {summary}]"
        ] + recent_messages
        
        print(f"\n💡 Historique résumé pour optimiser la mémoire.\n")

def reactive_agent_with_memory(user_input):
    """
    Agent avec mémoire conversationnelle et résumé automatique.
    """
    try:
        # Résumer si nécessaire
        summarize_old_messages()
        
        # Construire l'historique formaté
        history_text = "\n".join(conversation_messages)
        
        # Générer la réponse
        response = chain.invoke({"history": history_text, "input": user_input})
        
        # Sauvegarder dans l'historique
        conversation_messages.append(f"Humain: {user_input}")
        conversation_messages.append(f"Assistant: {response}")
        
        return response.strip()
    except Exception as e:
        return f"Une erreur s'est produite: {e}"

if __name__ == "__main__":
    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ['exit', 'quit']:
            break
        
        if not user_prompt.strip():
            continue

        if user_prompt.startswith("reset"):
            conversation_messages.clear()
            # Supprimer tous les documents de la collection
            try:
                # Récupérer tous les IDs et les supprimer
                all_docs = store.get()
                if all_docs['ids']:
                    store.delete(ids=all_docs['ids'])
                print("Historique de conversation et souvenirs réinitialisés.")
            except Exception as e:
                print(f"Erreur lors de la réinitialisation : {e}")
            continue

        if user_prompt.startswith("souviens-toi"):
            memory = user_prompt[len("souviens-toi"):].strip()
            if memory:
                # Sauvegarder dans le vector store
                store.add_texts([memory])
                print(f"Souvenir enregistré: {memory}")
            continue
        elif user_prompt.startswith("rappelle-moi"):
            reminder = user_prompt[len("rappelle-moi"):].strip()
            # Rechercher dans le vector store
            results = store.similarity_search(reminder, k=3)
            if results:
                print("Rappels trouvés:")
                for doc in results:
                    print(f"  - {doc.page_content}")
            else:
                print("Aucun rappel trouvé.")
            continue

        response = reactive_agent_with_memory(user_prompt)
        print(f"Agent: {response}")
