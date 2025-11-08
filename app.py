import streamlit as st
import os
import json
import io
import contextlib
from groq import Groq
from experta import *
from dotenv import load_dotenv


load_dotenv()


llm_model = None

#Fact Definitions

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
    priority = Field(int, default=99) # 99 is low priority (e.g., for LLM finds)
    description = Field(str, default="A popular tourist location.")
    
class Weather(Fact):
    """Holds the weather details for a location."""
    bad_region = Field(str, mandatory=True)
    month = Field(str, mandatory=True)

class ItineraryItem(Fact):
    """A final recommendation for the user."""
    stop_number = Field(int, mandatory=True) # e.g., "1", "2", "3"
    location = Field(str, mandatory=True)
    reason = Field(str, mandatory=True)
    description = Field(str, mandatory=True) 


class Warning(Fact):
    """A warning about a potential conflict or issue in the plan."""
    message = Field(str, mandatory=True)

class Recommendation(Fact):
    """An intermediate fact used for reasoning."""
    pass 

class FindInfo(Fact):
    """A fact to trigger the LLM to find info about an unknown interest."""
    interest = Field(str, mandatory=True)

class PotentialMatch(Fact):
    """A location that matches an interest, but is not yet in the final plan."""
    location = Field(str, mandatory=True)
    type = Field(str, mandatory=True)
    region = Field(str, mandatory=True)
    # --- NEW FIELDS ---
    priority = Field(int, mandatory=True)
    description = Field(str, mandatory=True)


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
            model="llama-3.1-8b-instant", 
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        
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

#Knowledge Engine

class ItineraryEngine(KnowledgeEngine):

    @DefFacts()
    def _initial_knowledge(self):
        # === 1. LOCATION FACTS (KNOWLEDGE BASE) ===
        # This is now structured by the "classic" tourist route.
        # Lower priority numbers = earlier in the logical trip.

        # --- Priority 1: Ancient Cities (Cultural Triangle) ---
        yield Location(name="Anuradhapura", type='history', region='cultural_triangle',
                       priority=1, description="Explore the vast ruins of the first ancient capital.")
        yield Location(name="Polonnaruwa", type='history', region='cultural_triangle',
                       priority=2, description="See the well-preserved medieval capital's temples and statues.")
        yield Location(name="Sigiriya", type='history', region='cultural_triangle',
                       priority=3, description="Climb the iconic Lion Rock fortress.")
        yield Location(name='Dambulla', type='history', region='cultural_triangle',
                       priority=4, description="Visit the impressive Golden Cave Temples.")

        # --- Priority 5: Hill Country ---
        yield Location(name='Kandy', type='culture', region='hill_country',
                       priority=5, description="Visit the Temple of the Tooth Relic and cultural shows.")
        yield Location(name='Nuwara Eliya', type='hiking', region='hill_country',
                       priority=6, description="Walk through lush tea plantations in 'Little England'.")
        yield Location(name="Horton Plains", type='hiking', region='hill_country',
                       priority=7, description="Hike to the stunning 'World's End' viewpoint.")
        yield Location(name='Ella', type='hiking', region='hill_country',
                       priority=8, description="See the Nine Arch Bridge and hike Little Adam's Peak.")

        # --- Priority 9: Wildlife (South) ---
        yield Location(name='Yala', type='wildlife', region='south_east',
                       priority=9, description="Go on a safari to spot leopards, elephants, and bears.")
        yield Location(name='Udawalawe', type='wildlife', region='south',
                       priority=10, description="See large herds of elephants at the National Park.")

        # --- Priority 11: South-West Coast ---
        yield Location(name='Mirissa', type='beach', region='south_west',
                       priority=11, description="Go whale watching (in season) or relax on the beach.")
        yield Location(name='Unawatuna', type='beach', region='south_west',
                       priority=12, description="A famous palm-lined beach with calm waters for swimming.")
        yield Location(name="Galle", type='culture', region='south_west',
                       priority=13, description="Walk the historic ramparts of the UNESCO Dutch Fort.")
        yield Location(name="Bentota", type='watersports', region='south_west',
                       priority=14, description="Popular hub for water skiing, jet skiing, and boat tours.")
        yield Location(name="Hikkaduwa", type='beach', region='south_west',
                       priority=15, description="Known for its coral reefs (snorkeling) and surf spots.")
        yield Location(name="Sinharaja", type='hiking', region='south_west',
                       priority=16, description="Trek in a UNESCO World Heritage rainforest, a biodiversity hotspot.")

        # --- Priority 20+: East Coast (Opposite weather season to South-West) ---
        yield Location(name='Arugam Bay', type='surfing', region='east_coast',
                       priority=20, description="A world-famous surfing destination for all levels.")
        yield Location(name="Trincomalee", type='beach', region='east_coast',
                       priority=21, description="Visit Koneswaram Temple and enjoy Nilaveli/Uppuveli beaches.")
        yield Location(name="Pasikudah", type='beach', region='east_coast',
                       priority=22, description="Famous for its long, shallow coastline, perfect for relaxing.")
        
        # --- Priority 30+: North (Culturally distinct trip) ---
        yield Location(name="Jaffna", type='culture', region='north',
                       priority=30, description="Explore the unique culture, temples, and islands of the Northern peninsula.")

        # === 2. WEATHER FACTS (KNOWLEDGE BASE) ===
        # This is a more complete model of the two main monsoons.

        # --- South-West Monsoon (Yala) ---
        # Approx May to September.
        # BAD for: south_west, south, hill_country
        # GOOD for: east_coast, north, cultural_triangle
        yield Weather(bad_region='south_west', month='may')
        yield Weather(bad_region='south_west', month='june')
        yield Weather(bad_region='south_west', month='july')
        yield Weather(bad_region='south_west', month='august')
        yield Weather(bad_region='south_west', month='september')
        
        yield Weather(bad_region='south', month='may')
        yield Weather(bad_region='south', month='june')
        yield Weather(bad_region='south', month='july')
        yield Weather(bad_region='south', month='august')

        yield Weather(bad_region='hill_country', month='may')
        yield Weather(bad_region='hill_country', month='june')
        yield Weather(bad_region='hill_country', month='july')
        yield Weather(bad_region='hill_country', month='august')


        # --- North-East Monsoon (Maha) ---
        # Approx October to February.
        # BAD for: east_coast, cultural_triangle, north
        # GOOD for: south_west, south, hill_country
        yield Weather(bad_region='east_coast', month='october')
        yield Weather(bad_region='east_coast', month='november')
        yield Weather(bad_region='east_coast', month='december')
        yield Weather(bad_region='east_coast', month='january')
        yield Weather(bad_region='east_coast', month='february')

        yield Weather(bad_region='cultural_triangle', month='october')
        yield Weather(bad_region='cultural_triangle', month='november')
        yield Weather(bad_region='cultural_triangle', month='december')
        yield Weather(bad_region='cultural_triangle', month='january')

        yield Weather(bad_region='north', month='october')
        yield Weather(bad_region='north', month='november')
        yield Weather(bad_region='north', month='december')
        yield Weather(bad_region='north', month='january')

    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Weather(bad_region=MATCH.region, month=MATCH.month),
        TEST(lambda i_list: i_list and 'beach' in i_list, MATCH.interests),
        salience=100
    )
    def determine_bad_weather_region(self, month, interests, region):
        # --- THIS IS THE CHANGE ---
        self.declare(Recommendation(
            avoid_region=region,
            reason=f"Avoiding {region} for beaches due to monsoon in {month}."
        ))
        
        self.declare(Warning(message=f"Avoiding {region} for beaches due to monsoon in {month}."))

    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests), # <-- It matches 'month' here
        Location(type='beach', region=MATCH.region),
        NOT(Recommendation(avoid_region=MATCH.region)),
        TEST(lambda i_list: i_list and 'beach' in i_list, MATCH.interests),
        salience=90
    )
    def determine_good_weather_region(self, region, month): # <-- Add 'month' to the signature
        if not any(isinstance(f, Recommendation) and f.get('suggest_region') == region for f in self.facts.values()):
            
            self.declare(Recommendation(
                suggest_region=region,
                reason=f"Good beach weather in {region} during {month} (not in monsoon)."
            ))
           

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

    @Rule(
        UserRequest(interests=MATCH.interests),
        # --- Grab the new fields ---
        Location(name=MATCH.name, type=MATCH.type, region=MATCH.region, 
                 priority=MATCH.p, description=MATCH.d),
        TEST(lambda interests, type: type in interests),
        NOT(Recommendation(avoid_region=MATCH.region)),
        salience=10
    )
    def find_potential_matches(self, interests, name, type, region, p, d):
        """
        Stage 1: Find all locations that match an interest and are not in a bad region.
        """
        # --- Pass the new fields along ---
        self.declare(PotentialMatch(location=name, type=type, region=region,
                                    priority=p, description=d))

    @Rule(
        UserRequest(duration=MATCH.d),
        TEST(lambda d: d < 10, MATCH.d),
        ItineraryItem(location='Sigiriya'), 
        ItineraryItem(location='Arugam Bay')
    )
    def conflict_travel_time_sigiriya_arugam(self):
        self.declare(Warning(message="High travel time between Cultural Triangle (Sigiriya) and East Coast (Arugam Bay). Difficult in < 10 days."))

    @Rule(
        UserRequest(duration=MATCH.d),
        ItineraryItem(location=MATCH.l1),
        ItineraryItem(location=MATCH.l2),
        ItineraryItem(location=MATCH.l3),
        TEST(lambda d: d < 7, MATCH.d), 
        TEST(lambda l1, l2, l3: l1 != l2 and l1 != l3 and l2 != l3, 
             MATCH.l1, MATCH.l2, MATCH.l3) 
    )
    def conflict_too_many_stops(self):
        if not any(isinstance(f, Warning) and "many stops" in f.get('message') for f in self.facts.values()):
            self.declare(Warning(message="Plan has many stops for a short trip. Consider focusing on one region."))


    @Rule(
        UserRequest(duration=MATCH.d),
        EXISTS(PotentialMatch()), 
        salience=-100
    )
    def build_final_itinerary(self, d):
        # ... (sections 0a, 0b, 1, 2 are all the same) ...
        # --- GATHER MATCHES and GATHER RECOMMENDATIONS ---
        
        # --- 0a. GATHER ALL MATCHES MANUALLY ---
        all_matches = []
        for f in self.facts.values():
            if isinstance(f, PotentialMatch):
                all_matches.append(f)
        
        if not all_matches:
            print("   > Planner: Fired but no PotentialMatch facts found.")
            return

        # --- 0b. GATHER OPTIONAL RECOMMENDATIONS MANUALLY ---
        avoid_region = None
        suggest_region = None
        for f in self.facts.values():
            if isinstance(f, Recommendation):
                if f.get('avoid_region'):
                    avoid_region = f.get('avoid_region')
                if f.get('suggest_region'):
                    suggest_region = f.get('suggest_region')
                    
        # --- 1. Define Pacing ---
        max_stops = max(1, (d // 2) + 1)

        # --- 2. Get Weather Recommendations ---
        print(f"   > Planner: Building plan. Max stops: {max_stops}.")
        if avoid_region:
            print(f"   > Planner: Avoiding {avoid_region}.")
        if suggest_region:
            print(f"   > Planner: Preferring {suggest_region}.")
            
        # --- 3. Filter and Sort Matches (NEW LOGIC) ---
        if avoid_region:
            filtered_matches = [m for m in all_matches if m['region'] != avoid_region]
        else:
            filtered_matches = all_matches
        
        # --- THIS IS THE NEW SORTING KEY ---
        def sort_key(match):
            # The base score is its route priority
            score = match['priority']
            
            # If it's in a "suggested" region, give it a big bonus
            # (We subtract 100 to make it go to the *front* of the list)
            if match['region'] == suggest_region:
                score -= 100
                
            return score
                
        sorted_matches = sorted(filtered_matches, key=sort_key)
        
        # --- 4. Select Final Stops (Unchanged) ---
        final_stops = []
        locations_added = set() 

        for match in sorted_matches:
            if len(final_stops) >= max_stops:
                break
                
            if match['location'] not in locations_added:
                final_stops.append(match)
                locations_added.add(match['location'])

        # --- 5. Declare Final Itinerary (UPGRADED) ---
        if not final_stops:
            print("   > Planner: Could not find any suitable stops.")
            return

        print(f"   > Planner: Selected {len(final_stops)} stops.")
        
        # We must re-sort the *final* list by priority to get the stop numbers right.
        # (This is in case the 'suggest_region' bonus made them out of order)
        final_sorted_stops = sorted(final_stops, key=lambda m: m['priority'])
        
        for i, stop in enumerate(final_sorted_stops):
            self.declare(ItineraryItem(
                stop_number = i + 1,
                location=stop['location'],
                reason=f"Matches '{stop['type']}' interest",
                description=stop['description']
            ))


#Helper Function
def run_itinerary_logic(duration, month, interests):
    """
    A helper function to run the expert system.
    This captures all console output and returns the final engine state.
    """
    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream):
        print("=" * 50)
        print(f"Generating Itinerary for: {month}, {duration} days, {interests}")
        print("---" * 10)

        engine = ItineraryEngine()
        engine.reset()

        engine.declare(UserRequest(duration=duration, month=month.lower(), interests=interests))

        print("--- Experta PASS 1: Detecting unknown interests...")
        engine.run()
        print("--- Experta PASS 1: Complete.")

        tasks = [f for f in engine.facts.values() if isinstance(f, FindInfo)]
        
        if tasks:
            print(f"\n--- LLM Agent: Found {len(tasks)} items to research...")
            for task in tasks:
                interest = task.get('interest')
                llm_data = call_llm_agent(interest)
                
                if llm_data:
                    engine.declare(Location(name=llm_data['name'],
                                            type=interest, 
                                            region=llm_data['region']))
            print("--- LLM Agent: Research complete.\n")

        print("--- Experta PASS 2: Re-running engine with new knowledge...")
        engine.run()
        print("--- Experta PASS 2: Complete.")

        
        print("\nEngine run finished.")
        print("---" * 10)

    log_output = log_stream.getvalue()
    return engine, log_output



st.set_page_config(page_title="Sri Lanka Itinerary Bot", layout="wide")
st.title("🇱🇰 Sri Lanka Itinerary Expert System")

# --- Sidebar for Inputs ---
with st.sidebar:

    st.header("✈️ Your Trip Details")
    
    # Trip Duration
    duration = st.slider("Trip Duration (days)", min_value=1, max_value=30, value=7)
    
    # Month
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    month = st.selectbox("Month of Travel", options=months)
    
    # Interests
    st.markdown("Enter interests, comma-separated (e.g. `beach, hiking, wildlife`)")
    interests_input = st.text_input(
        "Interests", 
        "hiking, history, rafting"
    )
    
    
    run_button = st.button("Generate Itinerary", use_container_width=True)


if run_button:

    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.sidebar.error("GROQ API Key is required!")
        st.stop()

    
    try:
        
        client = Groq(api_key=api_key)
        llm_model = client
        st.sidebar.success("Groq client configured.")
    except Exception as e:
        st.error(f"Error configuring Groq: {e}")
        st.stop()
    
    # 3. Process inputs
    interests_list = [i.strip().lower() for i in interests_input.split(',') if i.strip()]

    if not interests_list:
        st.error("Please enter at least one interest.")
        st.stop()

    # 4. Run the expert system
    with st.spinner("Running expert system... (This may take a moment if LLM is called)"):
        engine, log_output = run_itinerary_logic(duration, month, interests_list)

    st.header("Trip Plan Results")
    col1, col2 = st.columns(2)

    # 5. Display Warnings
    with col1:
        st.subheader("⚠️ Warnings")
        warnings = [f.get('message') for f in engine.facts.values() if isinstance(f, Warning)]
        if warnings:
            for w in set(warnings):
                st.warning(w)
        else:
            st.success("No conflicts found. Plan looks good!")

    with col2:
        st.subheader("🌴 Recommended Itinerary")
        
        # Get all ItineraryItem facts
        all_items = [f for f in engine.facts.values() if isinstance(f, ItineraryItem)]
        
        if all_items:
            # Sort them by the stop_number to ensure they are in order
            sorted_items = sorted(all_items, key=lambda x: x.get('stop_number'))
            
            item_data = []
            for i in sorted_items:
                item_data.append({
                    "Stop": i.get('stop_number'),
                    "Location": i.get('location'),
                    "Details": i.get('description'),
                    "Reason": i.get('reason')
                })
            
            # Set the column order for the dataframe
            st.dataframe(
                item_data,
                column_order=("Stop", "Location", "Details", "Reason"),
                use_container_width=True
            )
        else:
            st.info("No itinerary items could be generated for these preferences.")


    with st.expander("Show Full Execution Log"):
        st.code(log_output, language=None)