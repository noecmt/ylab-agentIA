from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict

# labo 2
# boucle de raisonnement -> action -> observation -> réflexion
def react_agent(question):
    reasoning = f"Je vais réfléchir à la question : {question}"
    print("🧠 Raisonnement:", reasoning)
    action = "appeler outil calendrier" if "calendrier" in question.lower() else "Répondre directement"
    print("⚙️ Action:", action)
    observation = "Vous avez une réunion à 15h" if action == "appeler outil calendrier" else "Aucune observation"
    reflection = f"Conclusion : {observation}"
    print("🔍 Réflexion:", reflection)
    return reflection

# labo 3 - Planner / Executor
def planner(goal):
    return ["Regarder disponibilités", "Analyser disponibilités", "Planifier réunion"]

def executor(plan):
    for step in plan:
        print("🧩 Étape :", step)

# labo 4 
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")

prompt_resume = ChatPromptTemplate.from_template(
    "Résume ce texte : {text}"
)

prompt_translate = ChatPromptTemplate.from_template(
    "Traduis le résumé en anglais : {resume}"
)

resume_chain = prompt_resume | llm | StrOutputParser()
translate_chain = prompt_translate | llm | StrOutputParser()

def run_workflow(text: str):
    resume = resume_chain.invoke({"text": text})
    traduction = translate_chain.invoke({"resume": resume})
    print("Résultat final :", traduction)
    return traduction

# labo 5
# Définir l'état du graphe
# Ici on fait un graphe simple avec 4 noeuds
class GraphState(TypedDict):
    query: str
    recherche: str
    synthese: str
    verification: str
    reponse: str

# Définir les noeuds du graphe
# chaque noeud prend et retourne un état de type GraphState
def recherche_node(state: GraphState) -> GraphState:
    print("🔍 Noeud Recherche")
    state["recherche"] = f"Résultats de recherche pour: {state['query']}"
    return state

def synthese_node(state: GraphState) -> GraphState:
    print("📝 Noeud Synthèse")
    state["synthese"] = f"Synthèse des résultats: {state['recherche']}"
    return state

def verification_node(state: GraphState) -> GraphState:
    print("✅ Noeud Vérification")
    state["verification"] = f"Vérification OK pour: {state['synthese']}"
    return state

def reponse_node(state: GraphState) -> GraphState:
    print("💬 Noeud Réponse")
    state["reponse"] = f"Réponse finale: {state['verification']}"
    return state

# Créer le graphe LangGraph
# liaison des noeuds
def create_langgraph():
    workflow = StateGraph(GraphState)
    
    # Ajouter les noeuds
    workflow.add_node("Recherche", recherche_node)
    workflow.add_node("Synthèse", synthese_node)
    workflow.add_node("Vérification", verification_node)
    workflow.add_node("Réponse", reponse_node)
    
    # Définir les arêtes (flux)
    workflow.set_entry_point("Recherche") # point d'entrée
    workflow.add_edge("Recherche", "Synthèse")
    workflow.add_edge("Synthèse", "Vérification")
    workflow.add_edge("Vérification", "Réponse")
    workflow.add_edge("Réponse", END) # point de sortie
    
    return workflow.compile()

# Semantic Kernel Planner (simulé)
# Un planificateur simple qui crée un plan basé sur un objectif
class SKPlanner:
    def create_plan(self, goal: str):
        print(f"📋 Création du plan pour: {goal}")
        return Plan([
            "Étape 1: Rechercher des informations sur la cybersécurité",
            "Étape 2: Analyser les menaces actuelles",
            "Étape 3: Identifier les meilleures pratiques",
            "Étape 4: Rédiger le rapport",
            "Étape 5: Réviser et finaliser"
        ])

# Classe Plan pour représenter le plan créé
class Plan:
    def __init__(self, steps):
        self.steps = steps
    
    def __repr__(self):
        return "\n".join([f"  - {step}" for step in self.steps])

def run_langgraph_demo():
    print("\n=== Démonstration LangGraph ===\n")
    graph = create_langgraph()
    result = graph.invoke({"query": "Cybersécurité et protection des données"})
    print(f"\n📊 Résultat final: {result['reponse']}\n")
    return result

def run_semantic_kernel_demo():
    print("\n=== Démonstration Semantic Kernel Planner ===\n")
    goal = "Créer un rapport sur la cybersécurité."
    planner = SKPlanner()
    plan = planner.create_plan(goal)
    print(f"Plan créé:\n{plan}\n")
    return plan


if __name__ == "__main__":
    # Labo 2 - ReAct Agent
    # user_query = "Quelle est la prochaine réunion dans mon calendrier ?"
    # react_agent(user_query)

    # user_query = "Quelle est la météo actuelle à Paris ?"
    # react_agent(user_query)

    # Labo 3 - Planner / Executor
    # goal = "Regarde les disponibilités et planifie une réunion avec l'équipe la semaine prochaine."
    # plan = planner(goal)
    # executor(plan)

    # Labo 4 - Workflow (chaînes séquentielles)
    # sample_text = "La cybersécurité est un domaine crucial dans le monde numérique actuel. Avec l'augmentation des cyberattaques, il est essentiel pour les entreprises de mettre en place des mesures de sécurité robustes pour protéger leurs données sensibles et assurer la continuité de leurs opérations."
    # run_workflow(sample_text)
    
    # Labo 5 - LangGraph et Semantic Kernel
    run_langgraph_demo()
    run_semantic_kernel_demo()