import os
import json
from groq import Groq  # <-- 1. ADDED THIS IMPORT
from experta import *

# --- 1. CONFIGURE LLM AGENT (Groq) ---

# We only configure Groq since it's the one that works.
llm_model = None
try:
    # Notice we're getting the GROQ_API_KEY now
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    llm_model = client  # We can just store the whole client
    
    print("LLM (Groq) configured successfully.")

except KeyError:
    print("="*50)
    print("ERROR: GROQ_API_KEY environment variable not set.")
    print("Please set the variable before running the script.")
    print("e.g., $env:GROQ_API_KEY = 'YOUR_API_KEY_HERE'")
    print("="*50)
except Exception as e:
    print(f"Error configuring Groq LLM: {e}")


# --- 2. DEFINITION OF CUSTOM FACT TYPES ---

# --- FIX: Added Fields to prevent NoneType errors ---

class UserRequest(Fact):
    """Holds the user's request details."""
    duration = Field(int, default=7)
    month = Field(str, mandatory=True)
    interests = Field(list, default=[]) 

class Location(Fact):
    """Holds the location details."""
    name = Field(str, mandatory=True)
    type = Field(str, mandatory=True)
    region = Field(str, mandatory=True)
    
class Weather(Fact):
    """Holds the weather details for a location."""
    bad_region = Field(str, mandatory=True)
    month = Field(str, mandatory=True)

class ItineraryItem(Fact):
    """A final recommendation for the user."""
    location = Field(str, mandatory=True)
    reason = Field(str, mandatory=True)

class Warning(Fact):
    """A warning about a potential conflict or issue in the plan."""
    message = Field(str, mandatory=True)

class Recommendation(Fact):
    """An intermediate fact used for reasoning."""
    pass # This fact is flexible (e.g., avoid_region, suggest_region)

class FindInfo(Fact):
    """A fact to trigger the LLM to find info about an unknown interest."""
    interest = Field(str, mandatory=True)


# --- 3. LLM AGENT FUNCTION ---

# --- FIX: Updated to use Groq methods and the correct model ---
def call_llm_agent(interest_to_find):
    """
    Calls the LLM to find the location for a specific interest.
    """
    if not llm_model:
        print(f"LLM not configured. Skipping search for '{interest_to_find}'.")
        return None
    
    system_prompt = f"""
    You are a research assistant for a Sri Lankan travel expert.
    Find the single best, most famous location in Sri Lanka for the following interest: "{interest_to_find}"

    Respond with ONLY a valid JSON object in the following format:
    {{"name": "LocationName", "region": "region_slug"}}

    Valid region_slugs are: 
    - cultural_triangle, hill_country, south_east, south, south_west, east_coast, north

    Example for "surfing":
    {{"name": "Arugam Bay", "region": "east_coast"}}
    
    If you cannot find a clear location, return:
    {{"name": null, "region": null}}
    """

    try:
        print(f"   > LLM Agent: Searching for a location for '{interest_to_find}'...")
        
        # --- THIS IS THE CORRECT GROQ CALL ---
        chat_completion = llm_model.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Find location for: {interest_to_find}"
                }
            ],
            # --- THIS IS THE CORRECT, WORKING MODEL ---
            model="llama-3.1-8b-instant", 
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}, # Ask for JSON
        )
        
        # --- THIS IS THE CORRECT GROQ RESPONSE ---
        json_text = chat_completion.choices[0].message.content

        data = json.loads(json_text)
        if data.get("name"):
            print(f"   > LLM Agent: Found {data['name']} in {data['region']}.")
            return data
        else:
            print(f"   > LLM Agent: Could not find a location for '{interest_to_find}'.")
            return None
            
    except Exception as e:
        print(f"   > LLM Agent: Error during API call - {e}")
        return None 

# --- 4. EXPERT SYSTEM KNOWLEDGE ENGINE ---

class ItineraryEngine(KnowledgeEngine):

    @DefFacts()
    def _initial_knowledge(self):
        #location data
        yield Location(name="Sigiriya", type='history', region='cultural_triangle')
        yield Location(name='Dambulla', type='history', region='cultural_triangle')
        yield Location(name='Kandy', type='culture', region='hill_country')
        yield Location(name='Ella', type='hiking', region='hill_country')
        yield Location(name='Nuwara Eliya', type='hiking', region='hill_country')
        yield Location(name='Yala', type='wildlife', region='south_east')
        yield Location(name='Udawalawe', type='wildlife', region='south')
        yield Location(name='Mirissa', type='beach', region='south_west')
        yield Location(name='Unawatuna', type='beach', region='south_west')
        
        #weather data
        #south_west monsoon
        yield Weather(bad_region='south_west', month='june')
        yield Weather(bad_region='south_west', month='july')
        yield Weather(bad_region='south_west', month='august')

        #North_east monsoon
        yield Weather(bad_region='east_coast', month='december')
        yield Weather(bad_region='east_coast', month='january')
        yield Weather(bad_region='cultural_triangle', month='december')

    # --- FIX: Moved 'salience' to the end and fixed 'TEST' ---
    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Weather(bad_region=MATCH.region, month=MATCH.month),
        # --- FIX: Check if i_list exists before checking 'in' ---
        TEST(lambda i_list: i_list and 'beach' in i_list, MATCH.interests),
        salience=100
    )
    def determine_bad_weather_region(self, month, interests, region):
        self.declare(Recommendation(avoid_region=region))
        self.declare(Warning(message=f"Avoiding {region} for beaches due to monsoon in {month}."))

    # --- FIX: Moved 'salience' to the end and fixed 'TEST' ---
    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Location(type='beach', region=MATCH.region),
        NOT(Recommendation(avoid_region=MATCH.region)),
        # --- FIX: Check if i_list exists before checking 'in' ---
        TEST(lambda i_list: i_list and 'beach' in i_list, MATCH.interests),
        salience=90
    )
    def determine_good_weather_region(self, region):
        if not any(isinstance(f, Recommendation) and f.get('suggest_region') == region for f in self.facts.values()):
            self.declare(Recommendation(suggest_region=region))

    # --- FIX: Moved 'salience' to the end ---
    @Rule(
        UserRequest(interests=MATCH.interests),
        salience=50
    )
    def detect_unknown_interests(self, interests):
        known_types = {f.get('type') for f in self.facts.values() if isinstance(f, Location)}
        
        for interest in interests:
            if interest not in known_types:
                print(f"   > Experta: Detected unknown interest: '{interest}'. Triggering agent.")
                self.declare(FindInfo(interest=interest))

    # --- FIX: Moved 'salience' to the end and fixed 'TEST' ---
    @Rule(
        
        UserRequest(interests=MATCH.interests),
        Location(name=MATCH.name, type=MATCH.type, region=MATCH.region),
        TEST(lambda interests, type: type in interests),
        NOT(Recommendation(avoid_region=MATCH.region)),
        salience=10
    )
    # --- THIS IS THE FIX ---
    def match_any_interest(self, interests, name, type, region):
        """
        Generic rule to match ANY interest...
        """
        self.declare(ItineraryItem(location=name, reason=f"Matches '{type}' interest"))
    
    
    # --- FIX: Fixed 'TEST' by passing 'd' ---
    @Rule(
        UserRequest(duration=MATCH.d),
        TEST(lambda d: d < 10, MATCH.d), # Only for short trips
        ItineraryItem(location='Sigiriya'), 
        ItineraryItem(location='Arugam Bay')
    )
    def conflict_travel_time_sigiriya_arugam(self):
        self.declare(Warning(message="High travel time between Cultural Triangle (Sigiriya) and East Coast (Arugam Bay). Difficult in < 10 days."))

    # --- FIX: Fixed 'TEST' by passing all variables ---
    @Rule(
        UserRequest(duration=MATCH.d),
        ItineraryItem(location=MATCH.l1),
        ItineraryItem(location=MATCH.l2),
        ItineraryItem(location=MATCH.l3),
        TEST(lambda d: d < 7, MATCH.d), # Only for very short trips
        TEST(lambda l1, l2, l3: l1 != l2 and l1 != l3 and l2 != l3, 
             MATCH.l1, MATCH.l2, MATCH.l3) 
    )
    def conflict_too_many_stops(self):
        if not any(isinstance(f, Warning) and "many stops" in f.get('message') for f in self.facts.values()):
            self.declare(Warning(message="Plan has many stops for a short trip. Consider focusing on one region."))

# --- 5. HELPER FUNCTION (no changes) ---

def get_itinerary(duration, month, interests):
    """
    A helper function to run the expert system.
    """
    print("=" * 50)
    print(f"Generating Itinerary for: {month}, {duration} days, {interests}")
    print("---" * 10)

    engine = ItineraryEngine()
    engine.reset()

    # This will now use the UserRequest class with default fields
    engine.declare(UserRequest(duration=duration, month=month.lower(), interests=interests))

    print("--- Experta PASS 1: Detecting unknown interests...")
    engine.run()
    print("--- Experta PASS 1: Complete.")

    tasks = [f for f in engine.facts.values() if isinstance(f, FindInfo)]
    
    if tasks:
        print(f"\n--- LLM Agent: Found {len(tasks)} items to research...")
        for task in tasks:
            interest = task.get('interest')
            # Call our new LLM function
            llm_data = call_llm_agent(interest)
            
            if llm_data:
                # Add the new knowledge from the LLM back into the engine
                engine.declare(Location(name=llm_data['name'],
                                        type=interest, # Use the original interest as the type
                                        region=llm_data['region']))
        print("--- LLM Agent: Research complete.\n")

    print("--- Experta PASS 2: Re-running engine with new knowledge...")
    engine.run()
    print("--- Experta PASS 2: Complete.")


    print("\n Warnings:")
    warnings = [f.get('message') for f in engine.facts.values() if isinstance(f, Warning)]
    if warnings:
        for w in set(warnings):
            print(f"   - {w}")
    else:
        print("   - No conflicts found. Plan looks good!")

    print("\n Recommended Itinerary Items:")
    items = [f for f in engine.facts.values() if isinstance(f, ItineraryItem)]
    if items:
            
        unique_locations = {i.get('location'): i for i in items}
        for i in unique_locations.values():
            print(f"   - Location: {i.get('location'):<15} | Reason: {i.get('reason')}")
    else:
        print("   - No itinerary items could be generated for these preferences.")
        
    print("---" * 10)


if __name__ == "__main__":

    get_itinerary(
        duration=7,
        month="August",  # Good weather for Arugam Bay
        interests=['beach', 'surfing', 'wildlife']
    )