from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 



# Initialiser le modèle Ollama
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")

# Initialiser le store de vecteurs avec HuggingFace Embeddings (plus fiable)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Stocker l'historique de conversation
conversation_history = []

# Template de prompt personnalisé
template = """Tu es un assistant IA serviable et amical. Utilise l'historique de conversation pour répondre de manière cohérente.

Historique de conversation:
{history}

Humain: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Créer la chaîne avec LangChain
chain = prompt | llm | StrOutputParser()

def reactive_agent_with_memory(user_input):
    """
    Agent avec mémoire conversationnelle utilisant LangChain.
    """
    try:
        # Construire l'historique formaté
        history_text = "\n".join([
            f"{'Humain' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        # Générer la réponse
        response = chain.invoke({"history": history_text, "input": user_input})
        
        # Sauvegarder dans l'historique
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
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



