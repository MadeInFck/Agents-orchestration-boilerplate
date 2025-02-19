import asyncio
import json
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import mistralai  # Assurez-vous que la bibliothèque mistralai est installée et configurée
from mistralai import Mistral

load_dotenv()
apikey = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=apikey)


# --- Interface for Specialized Agents ---
class Agent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    async def run(self, params: dict) -> dict:
        """
        Executes the task with the provided parameters and returns a result dictionary.
        """
        pass


# --- Specialized Agents using mistralai ---

class InternetSearchAgent(Agent):
    async def run(self, params: dict) -> dict:
        """
        Expected params:
        {
            "keywords": "openai chatgpt",
            "max_results": 5
        }
        In this simulation, we return a hard-coded result. Replace this with an actual API call.
        """
        keywords = params.get("keywords")
        max_results = params.get("max_results", 5)
        # Simulated search result data
        fake_data = {
            "items": [
                         {
                             "title": f"Sample Article about {keywords}",
                             "link": "https://example.com/article1",
                             "snippet": "This is a simulated snippet for article 1."
                         },
                         {
                             "title": f"Another Resource on {keywords}",
                             "link": "https://example.com/article2",
                             "snippet": "This is a simulated snippet for article 2."
                         }
                     ][:max_results]
        }
        return self.process_results(fake_data)

    def process_results(self, data: dict) -> dict:
        processed_results = []
        for item in data.get("items", []):
            processed_results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet")
            })
        return {"agent": self.name, "action": "search_internet", "results": processed_results}


class SummarizerAgent(Agent):
    async def run(self, params: dict) -> dict:
        text = params.get("text", "")
        if not text:
            return {"agent": self.name, "action": "summarize", "summary": ""}
        prompt = f"Provide a concise summary for the following text:\n\n{text}"
        response = await asyncio.to_thread(
            client.chat.complete,
            messages=[{"role": "user", "content": prompt}],
            model="open-mistral-nemo",
            max_tokens=500
        )
        summary = response.choices[0].message.content.strip()
        result = {"agent": self.name, "action": "summarize", "summary": summary}
        #print(f"[{self.name}] Summary generated.")
        return result


class EntityExtractorAgent(Agent):
    async def run(self, params: dict) -> dict:
        text = params.get("text", "")
        if not text:
            return {"agent": self.name, "action": "entity_extraction", "entities": []}
        prompt = (
            f"Extract named entities (persons, locations, organizations) from the following text:\n\n{text}\n\n"
            "Return only the entities separated by commas."
        )
        response = await asyncio.to_thread(
            client.chat.complete,
            messages=[{"role": "user", "content": prompt}],
            model="open-mistral-nemo",
            max_tokens=500
        )
        raw_entities = response.choices[0].message.content.strip()
        entities = [ent.strip() for ent in raw_entities.split(",") if ent.strip()]
        result = {"agent": self.name, "action": "entity_extraction", "entities": entities}
        #print(f"[{self.name}] Entities extracted.")
        return result


class TranslatorAgent(Agent):
    async def run(self, params: dict) -> dict:
        text = params.get("text", "")
        target_language = params.get("target_language", "en")
        if not text:
            return {"agent": self.name, "action": "translate", "translated_text": ""}
        prompt = f"Translate the following text into {target_language}:\n\n{text}"
        response = await asyncio.to_thread(
            client.chat.complete,
            messages=[{"role": "user", "content": prompt}],
            model="open-mistral-nemo",
            max_tokens=500
        )
        translated_text = response.choices[0].message.content.strip()
        result = {"agent": self.name, "action": "translate", "translated_text": translated_text}
        #print(f"[{self.name}] Text translated.")
        return result


# --- Final Formatter Agent ---
class FinalFormatterAgent(Agent):
    async def run(self, params: dict) -> dict:
        """
        Converts the aggregated JSON result (passed in params["aggregated"])
        into a natural language summary using the LLM.
        """
        aggregated = params.get("aggregated")
        if not aggregated:
            return {"agent": self.name, "action": "final_format", "formatted_text": "No result to format."}

        # Préparer le prompt pour transformer le JSON en texte lisible.
        prompt = (
            "Transform the following aggregated JSON result into a clear, well-structured summary in natural language:\n\n"
            f"{json.dumps(aggregated, ensure_ascii=False, indent=2)}\n\n"
            "Provide only the final text summary without additional commentary."
        )
        response = await asyncio.to_thread(
            client.chat.complete,
            messages=[{"role": "user", "content": prompt}],
            model="open-mistral-nemo",
            max_tokens=500
        )
        formatted_text = response.choices[0].message.content.strip()
        return {"agent": self.name, "action": "final_format", "formatted_text": formatted_text}


# --- Autonomous Dispatcher Agent ---
class AutonomousDispatcherAgent:
    """
    An autonomous dispatcher agent that analyzes a natural language request,
    converts it into an action plan (JSON), and delegates tasks to the specialized agents.
    """

    def __init__(self, agents: list):
        # Index the agents by their name for ease of access
        self.agents = {agent.name: agent for agent in agents}
        # Map an action to its corresponding agent
        self.action_to_agent = {
            "summarize": self.agents.get("summarizer"),
            "entity_extraction": self.agents.get("entity_extraction"),
            "translate": self.agents.get("translator"),
            "search_internet": self.agents.get("search_internet")
        }

    async def parse_natural_language_request(self, user_request: str) -> dict:
        """
        Converts the user's natural language request into a JSON structure describing tasks.
        A prompt is built for the LLM (here, mistralai) to output the minimal JSON format.
        """
        prompt = (
            "Analyze the following request and propose an action plan as a JSON structure "
            "using the following strict format:\n\n"
            '{\n  "tasks": [\n    { "action": "<action_name>", "params": { <parameters> } },\n    ...\n  ]\n}\n\n'
            f"Actions: {list(self.action_to_agent.keys())}\n\n"
            "Only return the JSON without any additional explanatory text.\n\n"
            "Request to analyze:\n"
            f"\"{user_request}\"\n"
        )
        response = await asyncio.to_thread(
            client.chat.complete,
            messages=[{"role": "user", "content": prompt}],
            model="open-mistral-nemo",
            max_tokens=500
        )
        json_text = response.choices[0].message.content.strip()
        #print("[Dispatcher] Raw LLM response for parsing:", json_text)
        try:
            plan = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("[Dispatcher] JSON decoding error. Falling back to an empty plan.")
            plan = {"tasks": []}
        return plan

    async def dispatch(self, user_request: str) -> dict:
        """
        Receives a natural language request, converts it into an action plan,
        executes the corresponding tasks, and returns the aggregated JSON result.
        """
        plan = await self.parse_natural_language_request(user_request)
        tasks = []
        for task in plan.get("tasks", []):
            action = task.get("action")
            params = task.get("params", {})
            # If the requested action does not have a corresponding agent, ignore the task
            agent = self.action_to_agent.get(action)
            if agent:
                tasks.append(agent.run(params))
            else:
                print(f"[Dispatcher] No matching agent found for action '{action}'.")

        results_list = await asyncio.gather(*tasks)
        # Aggregate results into a dictionary by action
        aggregated = {result["action"]: result for result in results_list if "action" in result}
        return aggregated


# --- Example Usage ---
async def main():
    # Create specialized agents
    summarizer = SummarizerAgent(name="summarizer")
    extractor = EntityExtractorAgent(name="entity_extraction")
    translator = TranslatorAgent(name="translator")
    search_internet = InternetSearchAgent(name="search_internet")
    final_formatter = FinalFormatterAgent(name="final_formatter")

    # Create the autonomous dispatcher agent with the list of specialized agents
    dispatcher = AutonomousDispatcherAgent(agents=[
        summarizer, extractor, translator, search_internet
    ])

    # Exemple de requêtes en langage naturel
    user_request1 = (
        "I want you to summarize the following text and extract its named entities: "
        "‘Steve and Mary are going to the theater tomorrow. The new Marvel is playing in the morning. "
        "They have made a reservation for the 10am session, and for food and drinks for a perfect shared time of happiness.’"
    )

    user_request2 = (
        "Translate this text into English: ‘Bonjour tout le monde!’"
    )

    user_request3 = (
        "Search internet for openai chatgpt documentation"
    )
    requests = [user_request1, user_request2, user_request3]

    for index,request in enumerate(requests):
        print(f"\n--- Processing User Request {index+1} ---")
        print("User request:\n", request)
        aggregated_result = await dispatcher.dispatch(request)
        #print("Aggregated result:\n", aggregated_result)
        # Utilisation du FinalFormatterAgent pour transformer le JSON en texte lisible
        final_text_result = await final_formatter.run({"aggregated": aggregated_result})
        print("Final formatted text:\n", final_text_result.get("formatted_text"))

if __name__ == "__main__":
    asyncio.run(main())
