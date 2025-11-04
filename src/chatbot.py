from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialiser le modèle Ollama
llm = Ollama(model="mistral", base_url="http://localhost:11434")

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
            
        response = reactive_agent_with_memory(user_prompt)
        print(f"Agent: {response}")