import os
from groq import Groq

def test_groq_api():
    """
    A simple function to test the Groq API connection and get a response.
    """
    try:
        # 1. Get the API key from your environment variables
        api_key = os.environ["GROQ_API_KEY"]
        
        # 2. Initialize the Groq client
        client = Groq(api_key=api_key)
        
        print("Groq client initialized. Sending a test message...")

        # 3. Send a test message
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Explain what Groq is in one simple sentence.",
                }
            ],
            # Use a fast, free, reliable model
            model="llama-3.1-8b-instant", 
            temperature=0.2, # Low temperature for a factual answer
        )
        
        # 4. Print the successful response
        response_text = chat_completion.choices[0].message.content
        print("\n--- TEST SUCCESSFUL ---")
        print(f"Groq Model Response: {response_text}")
        print("--------------------------")

    except KeyError:
        print("="*50)
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Please set the variable in your terminal before running:")
        print("  $env:GROQ_API_KEY = 'YOUR_API_KEY_HERE'")
        print("="*50)
    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(f"An error occurred: {e}")
        print("--------------------")

# This makes the script runnable
if __name__ == "__main__":
    test_groq_api()