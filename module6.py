import os
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from crewai import Agent, Crew, Task

# Désactiver la vérification OpenAI pour CrewAI
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-local-llm"

# labo 1
def agent_researcher(task: str) -> str:
    """Agent qui collecte les données"""

    # LLM avec binding des tools
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    
    # Messages
    messages = [
        SystemMessage(content="Tu es un assistant qui utilise ta connaissance pour collecter des informations, tu donnes des réponses courtes 5-6 lignes."),
        HumanMessage(content=task)
    ]
    
    # Appel au LLM
    response = llm.invoke(messages)
    return response.content

def agent_writer(data: str) -> str:
    """Agent qui synthétise les données"""

    # LLM avec binding des tools
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    
    # Messages
    messages = [
        SystemMessage(content="Tu es un assistant qui utilise ta connaissance pour synthétiser des informations en 2-3 phrases."),
        HumanMessage(content=data)
    ]
            
    # Appel au LLM
    response = llm.invoke(messages)
    return response.content

def agent_reviewer(text: str) -> str:
    """Agent qui améliore un texte donné"""

    # LLM avec binding des tools
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    
    # Messages
    messages = [
        SystemMessage(content="Tu es un assistant qui utilise ta connaissance pour améliorer un texte donné, tu fais un texte court."),
        HumanMessage(content=text)
    ]
            
    # Appel au LLM
    response = llm.invoke(messages)
    return response.content

def researcher(task: str) -> str:
    """Effectuer des recherches sur un sujet donné."""
    response = agent_researcher(task)
    return response

def writer(data: str) -> str:
    """Rédiger un texte basé sur les données fournies."""
    response = agent_writer(data)
    return response

def reviewer(text: str) -> str:
    """Relire et améliorer un texte donné."""
    response = agent_reviewer(text)
    return response

# labo 2
class Manager:
    """Classe pour gérer les agents."""

    def __init__(self, workers):
        self.workers = workers

    def run(self, task: str) -> str:
        """Exécuter une tâche en utilisant les agents disponibles."""
        print(f"\n Démarrage de la tâche: {task}")

        for worker in self.workers:
            print(f"\n Utilisation de l'agent: {worker.__name__}")
            task = worker(task)
            print(f"\n Résultat : {task}")
    
# labo 3
shared_memory = {}

class MemoryManager:
    """Classe pour gérer la mémoire partagée entre les agents."""

    def __init__(self, workers):
        self.memory = shared_memory
        self.workers = workers

    def get(self, key: str):
        return self.memory.get(key, None)

    def set(self, key: str, value):
        self.memory[key] = value

    def clear(self):
        self.memory.clear()

    def run(self, task: str) -> str:
        """Exécuter un agent avec accès à la mémoire partagée."""
        print(f"\n Démarrage de la tâche avec mémoire: {task}")

        for worker in self.workers:
            print(f"\n Utilisation de l'agent: {worker.__name__}")
            result = worker(task)
            task = result
            print(f"\n Résultat : {result}")
            self.set(worker.__name__, result)

        return result
    
# labo 4 - CrewAI : Orchestration multi-agents simplifiée avec Ollama
# CrewAI permet de coordonner plusieurs agents IA qui travaillent ensemble sur une tâche

# # 0. Configurer le LLM Ollama pour CrewAI
# ollama_llm = ChatOllama(
#     # model="llama3.1:latest",     # Nom complet avec tag
#     # base_url="http://127.0.0.1:11434",  # Utiliser 127.0.0.1 au lieu de localhost
#     # temperature=0.7
#     model="mistral", base_url="http://localhost:11434"
# )

# # 1. Créer des agents avec des rôles spécifiques et le LLM Ollama
# researcher_agent = Agent(
#     role="Researcher",           # Son rôle dans l'équipe
#     goal="Collecter des infos",  # Son objectif
#     backstory="Expert en recherche",  # Son contexte (optionnel)
#     llm=ollama_llm,              # IMPORTANT : Utiliser Ollama au lieu d'OpenAI
#     verbose=True
# )

# writer_agent = Agent(
#     role="Writer",
#     goal="Rédiger un résumé clair",
#     backstory="Rédacteur technique",
#     llm=ollama_llm,              # IMPORTANT : Utiliser Ollama
#     verbose=True
# )

# # 2. Définir une tâche simple
# task = Task(
#     description="Écrire un court texte sur Python",
#     agent=researcher_agent,      # Agent qui exécute la tâche
#     expected_output="Un texte de 3-4 lignes"  # Ce qu'on attend comme résultat
# )

# # 3. Créer une équipe (Crew) avec les agents
# crew = Crew(
#     agents=[researcher_agent, writer_agent],
#     tasks=[task],
#     verbose=True  # Affiche les étapes
# )

# # 4. Lancer l'exécution
# result = crew.kickoff() 
# print(f"Résultat CrewAI: {result}")

# labo 5 - Consensus entre agents : agréger plusieurs réponses
# Quand plusieurs agents donnent des réponses différentes, comment choisir la meilleure ?

# Simulation : 3 agents ont donné des réponses différentes
reponse1 = "Le meilleur lieu de vacances en été est la Côte d'Azur en France."
reponse2 = "La Bretagne est une excellente destination pour les vacances d'été."
reponse3 = "Les Alpes françaises offrent de magnifiques paysages pour les vacances d'été."

# Méthode 1 : Vote majoritaire
# Compte quelle réponse apparaît le plus souvent (utile si plusieurs agents donnent la même réponse)
def vote_majoritaire(responses: list) -> str:
    """Retourne la réponse la plus fréquente."""
    # max() trouve l'élément le plus fréquent en comptant les occurrences
    final = max(set(responses), key=responses.count)
    return final

# Méthode 2 : Agent arbitre
# Un agent IA analyse toutes les réponses et choisit la meilleure
def agent_arbitre(responses: list) -> str:
    """Un agent LLM analyse et choisit la meilleure réponse."""
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    
    # Préparer le prompt pour l'arbitre
    responses_text = "\n".join([f"- Réponse {i+1}: {r}" for i, r in enumerate(responses)])
    judge_prompt = f"Analyse ces réponses et choisis la meilleure (réponds avec le numéro et une brève justification) :\n{responses_text}"
    
    # L'agent arbitre décide
    messages = [
        SystemMessage(content="Tu es un juge qui analyse des réponses et choisis la meilleure. Sois concis."),
        HumanMessage(content=judge_prompt)
    ]
    
    result = llm.invoke(messages)
    return result.content

# Méthode 3 : Score de confiance
# Chaque réponse a un score, on prend celle avec le score le plus élevé
def score_confiance(responses_with_scores: list) -> dict:
    """Retourne la réponse avec le meilleur score de confiance."""
    # Trier par score décroissant et prendre la première
    weighted = sorted(responses_with_scores, key=lambda r: r['score'], reverse=True)[0]
    return weighted

# Exemples d'utilisation
def test_consensus():
    """Tester les 3 méthodes de consensus."""
    
    # Test 1 : Vote majoritaire (avec des doublons pour simuler un vote)
    print("=== Méthode 1 : Vote majoritaire ===")
    responses_vote = [reponse1, reponse2, reponse1, reponse3, reponse1]  # reponse1 apparaît 3 fois
    resultat_vote = vote_majoritaire(responses_vote)
    print(f"Réponse gagnante : {resultat_vote}\n")
    
    # Test 2 : Agent arbitre
    print("=== Méthode 2 : Agent arbitre ===")
    responses_arbitre = [reponse1, reponse2, reponse3]
    resultat_arbitre = agent_arbitre(responses_arbitre)
    print(f"Décision de l'arbitre : {resultat_arbitre}\n")
    
    # Test 3 : Score de confiance
    print("=== Méthode 3 : Score de confiance ===")
    responses_scores = [
        {'text': reponse1, 'score': 0.75},
        {'text': reponse2, 'score': 0.92},  # Score le plus élevé
        {'text': reponse3, 'score': 0.68}
    ]
    resultat_score = score_confiance(responses_scores)
    print(f"Meilleure réponse (score {resultat_score['score']}) : {resultat_score['text']}\n")

if __name__ == "__main__":
    # Tester les méthodes de consensus
    test_consensus()
    
    print("\n" + "="*50 + "\n")
    
    # Exemple d'utilisation des agents (labos précédents)
    research_topic = "Les meilleurs endroits à visiter en France en été."
    # print(f"Thème de recherche: {research_topic}")
    # research_data = researcher(research_topic)
    # print(f"Résultats bruts : {research_data}")
    # written_text = writer(research_data)
    # print(f"Texte rédigé : {written_text}")
    # reviewed_text = reviewer(written_text)
    # print(f"Texte relu et amélioré: {reviewed_text}")

    # manager = Manager(workers=[researcher, writer, reviewer])
    # final_result = manager.run(research_topic)
    # print(f"Résultat final: {final_result}")

    # memory_manager = MemoryManager(workers=[researcher, writer, reviewer])
    # memory_manager.run(research_topic)
    # for w in shared_memory:
    #     print(f"Clé mémoire: {w}, Valeur: {shared_memory[w]}")