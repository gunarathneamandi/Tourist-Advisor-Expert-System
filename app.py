import streamlit as st
import os
import json
import io
import contextlib
from groq import Groq
from experta import *

# --- 1. CONFIGURE LLM AGENT (Groq) ---
# This is now a global placeholder. 
# The Streamlit UI will configure it on-the-fly.
llm_model = None

# --- 2. DEFINITION OF CUSTOM FACT TYPES (Unchanged) ---

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
    pass 

class FindInfo(Fact):
    """A fact to trigger the LLM to find info about an unknown interest."""
    interest = Field(str, mandatory=True)


# --- 3. LLM AGENT FUNCTION (Unchanged) ---
# This function relies on the global `llm_model` variable, 
# which our Streamlit app will set.
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

# --- 4. EXPERT SYSTEM KNOWLEDGE ENGINE (Unchanged) ---

class ItineraryEngine(KnowledgeEngine):

    @DefFacts()
    def _initial_knowledge(self):
        yield Location(name="Sigiriya", type='history', region='cultural_triangle')
        yield Location(name='Dambulla', type='history', region='cultural_triangle')
        yield Location(name='Kandy', type='culture', region='hill_country')
        yield Location(name='Ella', type='hiking', region='hill_country')
        yield Location(name='Nuwara Eliya', type='hiking', region='hill_country')
        yield Location(name='Yala', type='wildlife', region='south_east')
        yield Location(name='Udawalawe', type='wildlife', region='south')
        yield Location(name='Mirissa', type='beach', region='south_west')
        yield Location(name='Unawatuna', type='beach', region='south_west')
        
        yield Weather(bad_region='south_west', month='june')
        yield Weather(bad_region='south_west', month='july')
        yield Weather(bad_region='south_west', month='august')
        yield Weather(bad_region='east_coast', month='december')
        yield Weather(bad_region='east_coast', month='january')
        yield Weather(bad_region='cultural_triangle', month='december')

    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Weather(bad_region=MATCH.region, month=MATCH.month),
        TEST(lambda i_list: i_list and 'beach' in i_list, MATCH.interests),
        salience=100
    )
    def determine_bad_weather_region(self, month, interests, region):
        self.declare(Recommendation(avoid_region=region))
        self.declare(Warning(message=f"Avoiding {region} for beaches due to monsoon in {month}."))

    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Location(type='beach', region=MATCH.region),
        NOT(Recommendation(avoid_region=MATCH.region)),
        TEST(lambda i_list: i_list and 'beach' in i_list, MATCH.interests),
        salience=90
    )
    def determine_good_weather_region(self, region):
        if not any(isinstance(f, Recommendation) and f.get('suggest_region') == region for f in self.facts.values()):
            self.declare(Recommendation(suggest_region=region))

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
        Location(name=MATCH.name, type=MATCH.type, region=MATCH.region),
        TEST(lambda interests, type: type in interests),
        NOT(Recommendation(avoid_region=MATCH.region)),
        salience=10
    )
    def match_any_interest(self, interests, name, type, region):
        self.declare(ItineraryItem(location=name, reason=f"Matches '{type}' interest"))
    
    @Rule(
        UserRequest(duration=MATCH.d),
        TEST(lambda d: d < 10, MATCH.d), # Only for short trips
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

# --- 5. MODIFIED HELPER FUNCTION ---
# This now returns the final engine object and the logs
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

        # The original print statements for results are removed.
        # We will extract the facts from the returned 'engine' object.
        print("\nEngine run finished.")
        print("---" * 10)

    log_output = log_stream.getvalue()
    return engine, log_output


# --- 6. STREAMLIT UI ---

st.set_page_config(page_title="Sri Lanka Itinerary Bot", layout="wide")
st.title("🇱🇰 Sri Lanka Itinerary Expert System")
st.markdown("This app uses an expert system and an LLM (Groq) to plan a trip.")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your GROQ_API_KEY", type="password")

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
    # 1. Validate API Key
    if not api_key:
        st.sidebar.error("GROQ API Key is required!")
        st.stop()

    # 2. Configure the global `llm_model` for the `call_llm_agent` function
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

    # 6. Display Itinerary
    with col2:
        st.subheader("🌴 Recommended Itinerary")
        items = [f for f in engine.facts.values() if isinstance(f, ItineraryItem)]
        
        if items:
            item_data = []
            unique_locations = {i.get('location'): i for i in items}
            for i in unique_locations.values():
                item_data.append({
                    "Location": i.get('location'),
                    "Reason": i.get('reason')
                })
            st.dataframe(item_data, use_container_width=True)
        else:
            st.info("No itinerary items could be generated for these preferences.")

    # 7. Display Logs
    with st.expander("Show Full Execution Log"):
        st.code(log_output, language=None)