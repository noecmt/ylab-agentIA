from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
from datetime import datetime

# Fichier de sauvegarde de la mémoire
MEMORY_FILE = "memory.json"

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

def save_memory():
    """
    Sauvegarde la mémoire dans un fichier JSON.
    """
    try:
        # Récupérer tous les souvenirs du vector store
        all_docs = store.get()
        memories = []
        
        if all_docs['ids']:
            for i, doc_id in enumerate(all_docs['ids']):
                memories.append({
                    "id": doc_id,
                    "content": all_docs['documents'][i],
                    "metadata": all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                })
        
        memory_data = {
            "last_saved": datetime.now().isoformat(),
            "conversation_history": conversation_messages[-10:],  # Garder seulement les 10 derniers
            "vector_store_memories": memories,
            "stats": {
                "total_messages": len(conversation_messages),
                "total_memories": len(memories)
            }
        }
        
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        print(f"Mémoire sauvegardée : {len(memories)} souvenirs, {len(conversation_messages)} messages")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
        return False

def load_memory():
    """
    Charge la mémoire depuis le fichier JSON.
    """
    global conversation_messages
    
    if not os.path.exists(MEMORY_FILE):
        print("Aucune mémoire précédente trouvée. Démarrage avec une mémoire vierge.")
        return False
    
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
        
        # Restaurer l'historique de conversation
        conversation_messages = memory_data.get("conversation_history", [])
        
        # Restaurer les souvenirs dans le vector store
        memories = memory_data.get("vector_store_memories", [])
        if memories:
            texts = [m["content"] for m in memories]
            metadatas = [m.get("metadata", {}) for m in memories]
            store.add_texts(texts, metadatas=metadatas)
        
        stats = memory_data.get("stats", {})
        last_saved = memory_data.get("last_saved", "inconnu")
        
        print(f"Mémoire chargée depuis {MEMORY_FILE}")
        print(f"   Dernière sauvegarde : {last_saved}")
        print(f"   Messages restaurés : {len(conversation_messages)}")
        print(f"   Souvenirs restaurés : {len(memories)}")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return False

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
        
        print(f"\nHistorique résumé pour optimiser la mémoire.\n")

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
    tour_eiffel_info = ("La tour Eiffel est une tour de fer puddlé de 330 mètres de hauteur, avec antennes, située à Paris, à l extrémité nord-ouest du parc du Champ-de-Mars en bordure de la Seine dans le 7e arrondissement. Son adresse officielle est 5, avenue Anatole-France. Construite en deux ans par Gustave Eiffel et ses collaborateurs pour l Exposition universelle de Paris de 1889, célébrant le centenaire de la Révolution française, et initialement nommée « tour de 300 mètres », elle est devenue le symbole de la capitale française et un site touristique de premier plan : il s’agit du quatrième site culturel français payant le plus visité en 2016, avec 5,9 millions de visiteurs. Depuis son ouverture au public, elle a accueilli plus de 300 millions de visiteurs.")
    store.add_texts([tour_eiffel_info])
    print("Souvenir initial ajouté : Informations sur la tour Eiffel.\n")
    
    # Charger la mémoire au démarrage
    print("="*60)
    print("Chatbot avec mémoire persistante")
    print("="*60)
    load_memory()
    print("\nCommandes disponibles:")
    print("  • souviens-toi <info> : enregistrer un souvenir")
    print("  • rappelle-moi <query> : chercher dans les souvenirs")
    print("  • reset : réinitialiser la mémoire")
    print("  • save : sauvegarder manuellement")
    print("  • exit/quit : quitter (sauvegarde automatique)")
    print("="*60 + "\n")
    
    try:
        while True:
            user_prompt = input("You: ")
            if user_prompt.lower() in ['exit', 'quit']:
                print("\nSauvegarde de la mémoire avant de quitter...")
                save_memory()
                print("Au revoir !")
                break
            
            if not user_prompt.strip():
                continue
            
            # Commande manuelle de sauvegarde
            if user_prompt.lower() == "save":
                save_memory()
                continue

            if user_prompt.startswith("reset"):
                conversation_messages.clear()
                # Supprimer tous les documents de la collection
                try:
                    # Récupérer tous les IDs et les supprimer
                    all_docs = store.get()
                    if all_docs['ids']:
                        store.delete(ids=all_docs['ids'])
                    print("Historique et souvenirs réinitialisés.")
                    # Supprimer le fichier de sauvegarde
                    if os.path.exists(MEMORY_FILE):
                        os.remove(MEMORY_FILE)
                        print("Fichier de sauvegarde supprimé.")
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
    
    except KeyboardInterrupt:
        print("\n\nInterruption détectée. Sauvegarde de la mémoire...")
        save_memory()
        print("Au revoir !")
    except Exception as e:
        print(f"\nErreur : {e}")
        print("Tentative de sauvegarde de la mémoire...")
        save_memory()
