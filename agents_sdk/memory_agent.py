import openai
import os
import time

# Set your OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = "asst_9J00Lf74XCufZBXOhd8T4ddi"

def create_thread():
    return client.beta.threads.create()

def send_message(thread_id, user_input):
    # Add user message to thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
    thread_id=thread_id,
    assistant_id=ASSISTANT_ID,
    instructions="You are a memory-aware emotional companion. Read the full thread and adapt your responses based on how the user's feelings evolve over time."
    )


    # Wait for run to complete
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )

        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            print("🚨 Run failed with error:")
            print(run_status.last_error)
            return "❌ Assistant run failed."

        time.sleep(1)

    # Get assistant's latest reply
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    # for message in reversed(messages.data):
    #     if message.role == "assistant":
    #         return message.content[0].text.value

    assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
    assistant_messages.sort(key=lambda m: m.created_at, reverse=True)
    return assistant_messages[0].content[0].text.value if assistant_messages else "🤖 No assistant response found."
    
    

    return "🤖 No assistant response found."


if __name__ == "__main__":
    print("🧠 Memory Agent — Assistant SDK")
    thread = create_thread()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = send_message(thread.id, user_input)
        # # DEBUG: Show last few thread messages
        # print("🧵 Thread history:")
        # messages = client.beta.threads.messages.list(thread_id=thread.id)
        # for msg in reversed(messages.data[-5:]):
        #     print(f"{msg.role}: {msg.content[0].text.value}")
        
        print(f"Assistant: {reply}\n")