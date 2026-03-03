import os
import random
from typing import List, Dict, Any

# Intégration Ollama (même approche que dans module6.py)
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatOllama = None
    SystemMessage = lambda content: type("Sys", (), {"content": content})
    HumanMessage = lambda content: type("Hum", (), {"content": content})

os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-local-llm"


def _make_llm(model: str = "mistral", base_url: str = "http://localhost:11434"):
    if ChatOllama is None:
        return None
    return ChatOllama(model=model, base_url=base_url)


def _invoke_llm(prompt: str) -> str:
    llm = _make_llm()
    messages = [SystemMessage(content="Tu es un assistant concis."), HumanMessage(content=prompt)]
    if llm is None:
        return f"[SIMULÉ] {prompt[:80]}..."
    try:
        res = llm.invoke(messages)
        return getattr(res, "content", str(res))
    except Exception as e:
        return f"[ERREUR LLM] {e}"


# --------------------- Labo 1: Identifier les Design Patterns ------------
def lab1_identify_patterns(use_cases: List[str]) -> List[Dict[str, str]]:
    """Classifie des use-cases simples en patterns (FAQ, Mémoire, RAG, Multi-Agent).

    Retourne une liste de dicts: {'use_case','pattern','justification'}.
    """
    mapping = {
        "Chatbot FAQ": ("Réflexe", "Réponse directe, faible contexte"),
        "Assistant personnel": ("Mémoire", "Stocke le contexte et l'historique utilisateur"),
        "Chat PDF": ("RAG", "Recherche dans un index documentaire"),
        "Équipe collaborative": ("Multi-Agent", "Coordination de rôles")
    }
    results = []
    for uc in use_cases:
        pat, just = mapping.get(uc, ("Inconnu", "À analyser"))
        results.append({"use_case": uc, "pattern": pat, "justification": just})
    return results


# --------------------- Labo 2: Associer Patterns et Frameworks ----------
def lab2_associate_frameworks(patterns: List[str]) -> List[Dict[str, str]]:
    """Associe chaque pattern à un framework recommandé (simulé).

    Retourne {'pattern','framework','raison'}.
    """
    fav = {
        "RAG": ("LlamaIndex", "Indexation documentaire et recherche vectorielle"),
        "Planner": ("Semantic Kernel", "Planification dynamique et skills"),
        "Multi-Agent": ("CrewAI", "Coordination multi-agents"),
        "Mémoire": ("LangChain", "Gestion d'état et mémoire conversationnelle"),
        "Réflexe": ("LangChain", "Réponses directes et prompt templates")
    }
    out = []
    for p in patterns:
        fw, reason = fav.get(p, ("Autre", "Choix à justifier"))
        out.append({"pattern": p, "framework": fw, "raison": reason})
    return out


# --------------------- Labo 3: Créer un Agent Hybride (RAG + Multi-Agent) -
def lab3_hybrid_agent_scaffold(documents: List[str], goal: str) -> Dict[str, Any]:
    """Scaffold montrant comment on pourrait combiner LlamaIndex (RAG)
    et CrewAI (coordination). On ne requiert pas que ça fonctionne.
    """
    # Placeholders / imports simulés
    try:
        from llama_index import SimpleVectorIndex  # type: ignore
    except Exception:
        SimpleVectorIndex = None

    try:
        from crewai import Agent, Crew, Task  # type: ignore
    except Exception:
        Agent = Crew = Task = None

    log = {"goal": goal, "documents_count": len(documents), "steps": []}

    # RAG step (simulé)
    if SimpleVectorIndex is None:
        log["steps"].append({"step": "RAG", "note": "[SIMULÉ] index creation"})
    else:
        # example (non-executable placeholder)
        index = SimpleVectorIndex.from_texts(documents)
        log["steps"].append({"step": "RAG", "note": "index created", "index": str(type(index))})

    # CrewAI coordination (simulé)
    if Crew is None:
        log["steps"].append({"step": "CrewAI", "note": "[SIMULÉ] create agents and kickoff"})
    else:
        researcher = Agent(role="Researcher", goal="Interroger le RAG")
        writer = Agent(role="Writer", goal="Synthétiser la réponse")
        crew = Crew(agents=[researcher, writer])
        # crew.kickoff(goal)  # commented: illustrative only
        log["steps"].append({"step": "CrewAI", "note": "agents created (non exécuté)"})

    # Return a small orchestration spec that looks plausible
    orchestration = {
        "pattern": "RAG+Multi-Agent",
        "flow": ["index -> researcher -> writer -> aggregate"],
        "log": log
    }
    return orchestration


# --------------------- Labo 4: Orchestration Hybride ---------------------
def lab4_orchestration_simulation(goal: str, plan: List[str]) -> Dict[str, Any]:
    """Simule une orchestration hybride (Planner + CrewAI + LangGraph).

    Retourne un dict décrivant les étapes et états (plausible, non-exécutable).
    """
    states = []
    for step in plan:
        states.append({"step": step, "status": "pending", "assigned_to": "role_placeholder"})

    # Simuler transition
    for i, s in enumerate(states):
        s["status"] = "completed"
        s["assigned_to"] = "Researcher" if i == 0 else ("Writer" if i == 1 else "Validator")

    graph = {
        "goal": goal,
        "states": states,
        "langgraph_spec": "(simulé) nodes: Collecte -> Synthèse -> Validation"
    }
    return graph


# --------------------- Labo 5: Étude prospective -------------------------
def lab5_prospective_study(year: int = 2027) -> Dict[str, Any]:
    """Retourne une vision architecturale synthétique pour un agent futuriste.

    Fournit architecture, frameworks combinés et mécanismes de gouvernance.
    """
    arch = {
        "year": year,
        "architecture": "Hybrid edge-cloud: local LLMs + cloud RAG + governance layer",
        "frameworks": ["LangChain", "LlamaIndex", "CrewAI", "Semantic Kernel", "LangGraph"],
        "governance": {
            "safety_layer": "policy engine + approval workflows",
            "auditing": "immutable logs + external verifier (SelfCheck)"
        },
        "short_pitch": f"Agent autonome 2026-{year}: hybride, vérifiable, et gouverné"
    }
    return arch


if __name__ == "__main__":
    print("\n--- MODULE 8 : Design Patterns & Frameworks Avancés (démo) ---\n")

    # Labo 1 demo
    use_cases = ["Chatbot FAQ", "Assistant personnel", "Chat PDF", "Équipe collaborative"]
    p1 = lab1_identify_patterns(use_cases)
    print("Labo1 - Patterns identifiés:")
    for r in p1:
        print(f" - {r['use_case']}: {r['pattern']} ({r['justification']})")

    # Labo 2 demo
    patterns = [r["pattern"] for r in p1]
    p2 = lab2_associate_frameworks(patterns)
    print("\nLabo2 - Frameworks associés:")
    for r in p2:
        print(f" - {r['pattern']} -> {r['framework']} : {r['raison']}")

    # Labo 3 demo (scaffold)
    docs = ["Doc 1: tendances IA", "Doc 2: rapport 2026"]
    h = lab3_hybrid_agent_scaffold(docs, goal="Analyse des tendances IA 2025")
    print("\nLabo3 - Hybrid scaffold:")
    print(h["pattern"], h["flow"])

    # Labo 4 demo
    plan = ["Collecte", "Synthèse", "Validation"]
    orches = lab4_orchestration_simulation("Rédiger une note sur l'énergie verte.", plan)
    print("\nLabo4 - Orchestration states:")
    for s in orches["states"]:
        print(f" - {s['step']}: {s['status']} (role={s['assigned_to']})")

    # Labo 5 demo
    vis = lab5_prospective_study(2027)
    print("\nLabo5 - Vision prospective:")
    print(vis["short_pitch"]) 

    print("\n--- Fin demo Module 8 ---\n")
