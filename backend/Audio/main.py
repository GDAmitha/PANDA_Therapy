from transcribe_to_json import transcribe_audio_to_json
from assign_speaker_roles import assign_speaker_roles
from assign_audio_emotions import assign_audio_emotions
from patient_chatbot import patient_chat
from letta_client import Letta
import os
from models import Patient
client = Letta(token=os.getenv("LETTA_API_KEY"))


class User:
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender
        self.phone_number = None
        self.email = None
        self.address = None
    
    def login(self):
        print(f"{self.name} has logged in.")

    def logout(self):
        print(f"{self.name} has logged out.")

    def load_therapy_session_audio(self, audio_path):
        print(f"{self.name} is loading audio from {audio_path}.")
        success, result = transcribe_audio_to_json("/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav")

        assign_speaker_roles(result, "Nate", "Dimple")
        assign_audio_emotions("/Users/natedamstra/Downloads/Mock Therapy Convo with Dimple.wav", "speaker_assign_transcript/Mock Therapy Convo with Dimple_transcript_assigned.json")


    def __str__(self):
        return f"{self.name} ({self.age}, {self.gender})"  



class Therapist(User):
    def __init__(self, name, age, gender, agent_id):
        super().__init__(name, age, gender)



class Patient(User):
    def __init__(self, name, age, gender, agent_id):
        super().__init__(name, age, gender)
        self.agent_id = agent_id
        self.therapist_id = None

    def chat_with_agent(self, message: str):
        patient_chat(message, self)

    def chat_with_therapist(self, message: str):
        print(f"{self.name} is chatting with the therapist.")

        

# client.agents.templates.migrate(agent_id="agent-d79dcaed-2dc3-4ae1-b691-73ffa10145da", to_template="Panda_Therapist:latest", preserve_core_memories=True)

# patient = Patient("Dimple", 25, "Female")
therapist = Therapist("Dr. Nat", 25, "Male", "agent-d79dcaed-2dc3-4ae1-b691-73ffa10145da")

nate = Patient("Nate", 21, "Male", "agent-7b8ebf16-bc97-4919-ad4a-18637ff51eab")



nate.chat_with_agent("How can you help me with stress management?")


