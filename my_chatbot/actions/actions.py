# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionProvideDiagnosis(Action):
    def name(self) -> Text:
        return "action_provide_diagnosis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        symptoms = [e['value'].lower() for e in tracker.latest_message.get("entities", []) if e['entity'] == 'symptom']

        if not symptoms:
            diagnosis = "I couldn't detect any symptoms. Can you describe how you're feeling in more detail?"
        elif "fever" in symptoms and "cough" in symptoms:
            diagnosis = "You may have a viral infection like the flu or COVID-19."
        elif "headache" in symptoms and "nausea" in symptoms:
            diagnosis = "It could be a migraine."
        elif "chest pain" in symptoms:
            diagnosis = "This could be serious. Please seek immediate medical help."
        else:
            diagnosis = "I need more symptoms to give a better guess."

        dispatcher.utter_message(text=diagnosis)
        return []
