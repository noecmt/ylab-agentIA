import requests
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_fixed

# utils
API_KEY = ""

# labo 1
def get_weather(city):
    return f"Il fait 23° degré à {city}."

# labo 2 - Intégrer une API réelle
@tool
def get_weather_api(city: str) -> str:
    """Obtenir la météo actuelle d'une ville."""
    if not API_KEY or API_KEY == "METS_TA_CLE_ICI":
        return {"error": "Clé API manquante. Inscris-toi sur openweathermap.org"}
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=fr" 
    data = requests.get(url).json() 
    
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
        
    return f"À {city}: {weather}, {temp}°C (ressenti {feels_like}°C), humidité {humidity}%"

# labo 3
@tool
def get_news(topic: str) -> str:
    """Lire les dernières actualités sur un sujet."""
    return f"Voici les dernières nouvelles sur {topic}."

@tool
def get_time(city: str) -> str:
    """Obtenir l'heure locale d'une ville."""
    return f"Il est 15h00 à {city}."

def run_agent(user_query: str):
    """Agent qui utilise les outils automatiquement."""
    
    # Liste des outils disponibles (objets tools, pas des dicts)
    tools = [get_weather_api, get_news, get_time]

    # LLM avec binding des tools
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    llm_with_tools = llm.bind_tools(tools)
    
    # Messages
    messages = [
        SystemMessage(content="Tu es un assistant qui utilise des outils pour répondre aux questions."),
        HumanMessage(content=user_query)
    ]
    
    print(f"\n Question: {user_query}")
    
    # Appel au LLM
    response = llm_with_tools.invoke(messages)
    
    # Vérifier si le LLM veut appeler un outil
    if response.tool_calls:
        print(f"Outil appelé: {response.tool_calls[0]['name']}")
        print(f"Arguments: {response.tool_calls[0]['args']}")
        
        # Exécuter l'outil
        tool_name = response.tool_calls[0]['name']
        tool_args = response.tool_calls[0]['args']
        
        # Trouver et exécuter l'outil
        tool_map = {t.name: t for t in tools}
        result = tool_map[tool_name].invoke(tool_args)
        
        print(f"Résultat: {result}")
        
        # Reformuler la réponse avec le résultat
        messages.append(response)
        messages.append(HumanMessage(content=f"Résultat de l'outil: {result}. Reformule une réponse naturelle."))
        final_response = llm.invoke(messages)
        print(f"\nRéponse finale: {final_response.content}")
    else:
        print(f"Réponse directe: {response.content}")

# labo 4 - Réponse au format JSON
# schéma de la réponse JSON
json_response_schema = {
  "time" : "",
  "tool_to_use" : "",
  "args" : "",
  "result_tool" : "",
  "response_llm" : ""
}

def run_agent2(user_query: str):
    """Agent qui utilise des outils automatiquement et répond au format JSON."""
    from datetime import datetime
    
    # Liste des outils disponibles (objets tools, pas des dicts)
    tools = [get_weather_api, get_news, get_time]

    # LLM avec binding des tools
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434", format="json")
    llm_with_tools = llm.bind_tools(tools)
    
    # Messages
    messages = [
        SystemMessage(content=f"""Tu es un assistant qui utilise des outils pour répondre aux questions.
Tu DOIS répondre UNIQUEMENT au format JSON valide suivant (sans texte avant/après):
{json.dumps(json_response_schema, indent=2)}

Remplis chaque champ:
- time: timestamp actuel
- tool_to_use: nom de l'outil appelé
- args: arguments passés à l'outil
- result_tool: résultat brut de l'outil
- response_llm: ta réponse en langage naturel basée sur le résultat"""),
        HumanMessage(content=user_query)
    ]
    
    print(f"\n Question: {user_query}")
    
    # Appel au LLM
    response = llm_with_tools.invoke(messages)
    
    # Vérifier si le LLM veut appeler un outil
    if response.tool_calls:
        tool_name = response.tool_calls[0]['name']
        tool_args = response.tool_calls[0]['args']
        
        print(f"Outil appelé: {tool_name}")
        print(f"Arguments: {tool_args}")
        
        # Trouver et exécuter l'outil
        tool_map = {t.name: t for t in tools}
        result = tool_map[tool_name].invoke(tool_args)
        
        print(f"Résultat: {result}")
        
        # Construire la réponse JSON
        messages.append(response)
        messages.append(HumanMessage(content=f"""Résultat de l'outil: {result}
        
Réponds UNIQUEMENT avec un JSON valide suivant ce schéma exact:
{{
  "time": "{datetime.now().isoformat()}",
  "tool_to_use": "{tool_name}",
  "args": {json.dumps(tool_args)},
  "result_tool": "{result}",
  "response_llm": "ta réponse en français ici"
}}

JSON uniquement, pas de texte avant/après:"""))
        
        final_response = llm.invoke(messages)
        print(f"\nRéponse JSON:\n{final_response.content}")
        
        # Parser et afficher proprement
        try:
            parsed = json.loads(final_response.content)
            print(f"\nJSON valide:\n{json.dumps(parsed, indent=2, ensure_ascii=False)}")
        except:
            print("Réponse pas en JSON valide")
    else:
        print(f"Réponse directe: {response.content}")

# labo 5 - Gestion d'erreurs
@tool
def get_weather_errorAPI(param: str) -> str:
    """Obtenir la météo actuelle d'une ville avec gestion d'erreurs."""
    raise Exception("Erreur API") 

@tool
def fallback() -> str:
    """Outil de secours en cas d'erreur.""" 
    return f"Impossible d'obtenir la réponse de l'outil, réessaie plus tard."

def run_agent3(user_query: str):
    """Agent qui utilise des outils avec gestion d'erreurs."""
    
    # Liste des outils disponibles (objets tools, pas des dicts)
    tools = [get_weather_errorAPI, fallback, get_news, get_time]

    # LLM avec binding des tools
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    llm_with_tools = llm.bind_tools(tools)
    
    # Messages
    messages = [
        SystemMessage(content="Tu es un assistant qui utilise des outils pour répondre aux questions. Gère les erreurs des outils."),
        HumanMessage(content=user_query)
    ]
    
    print(f"\n Question: {user_query}")
    
    # Appel au LLM
    response = llm_with_tools.invoke(messages)
    
    # Vérifier si le LLM veut appeler un outil
    if response.tool_calls:
        tool_name = response.tool_calls[0]['name']
        tool_args = response.tool_calls[0]['args']
        
        print(f"Outil appelé: {tool_name}")
        print(f"Arguments: {tool_args}")
        
        # Trouver et exécuter l'outil
        tool_map = {t.name: t for t in tools}
        
        try:
            result = tool_map[tool_name].invoke(tool_args)
            print(f"Résultat: {result}")
            
            # Reformuler la réponse avec le résultat
            messages.append(response)
            messages.append(HumanMessage(content=f"Résultat de l'outil: {result}. Reformule une réponse naturelle."))
            final_response = llm.invoke(messages).content
        except Exception as e:
            print(f"Erreur outil: {e}")
            # Fallback
            result = fallback().invoke({})
            print(f"Fallback: {result}")
            final_response = result
        
        print(f"\nRéponse finale: {final_response}")
    else:
        print(f"Réponse directe: {response.content}")   

if __name__ == "__main__":
    #print(get_weather_api("Paris"))

    #run_agent("Quelle est la météo à Lyon ?")
    #run_agent("Quelles sont les dernières nouvelles sur la technologie ?")
    #run_agent("Quelle heure est-il à New York ?")

    run_agent3("Quelle est la météo à Lyon ?")