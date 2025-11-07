import os
import json
import streamlit as st
from groq import Groq  # <-- Using Groq
from experta import *
from contextlib import contextmanager, redirect_stdout
from io import StringIO


@st.cache_resource
def get_groq_client():
    """
    Creates and returns a cached Groq client instance.
    Handles API key errors and displays them in the Streamlit UI.
    """
    try:
        
        api_key = os.environ["GROQ_API_KEY"]
        if not api_key:
            raise KeyError  
            
        client = Groq(api_key=api_key)
        print("LLM (Groq) client configured successfully.")
        return client
    
    except KeyError:
        
        st.error("ERROR: GROQ_API_KEY environment variable not set.")
        st.code("Please set it in your terminal before running:\n\n$env:GROQ_API_KEY = 'YOUR_API_KEY_HERE'")
        return None
    except Exception as e:
        st.error(f"Error configuring Groq LLM: {e}")
        return None

# Get the cached client.
llm_model = get_groq_client()


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


@st.cache_data
def call_llm_agent(interest_to_find):
    """
    Calls the Groq LLM to find the location for a specific interest.
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

@st.cache_resource
def get_knowledge_engine():
    """
    Defines and returns a cached instance of the ItineraryEngine.
    """
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
            yield Weather(bad_region='south_west', month='june')
            yield Weather(bad_region='south_west', month='july')
            yield Weather(bad_region='south_west', month='august')
            yield Weather(bad_region='east_coast', month='december')
            yield Weather(bad_region='east_coast', month='january')
            yield Weather(bad_region='cultural_triangle', month='december')
    
       
        
        @Rule(
            UserRequest(month=MATCH.month, interests=MATCH.interests),
            Weather(bad_region=MATCH.region, month=MATCH.month),
            # The lambda automatically gets the 'interests' variable
            TEST(lambda interests: interests and ('beach' in interests or 'surfing' in interests)),
            salience=100
        )
        def determine_bad_weather_region(self, month, interests, region):
            self.declare(Recommendation(avoid_region=region))
            self.declare(Warning(message=f"Avoiding {region} for beaches/surfing due to monsoon in {month}."))
    
        @Rule(
            UserRequest(month=MATCH.month, interests=MATCH.interests),
            Location(type='beach', region=MATCH.region),
            NOT(Recommendation(avoid_region=MATCH.region)),
            TEST(lambda interests: interests and ('beach' in interests or 'surfing' in interests)),
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
            TEST(lambda interests, type: type in interests), # This lambda is correct
            NOT(Recommendation(avoid_region=MATCH.region)),
            salience=10
        )
        def match_any_interest(self, interests, name, type, region):
            self.declare(ItineraryItem(location=name, reason=f"Matches '{type}' interest"))
        
        @Rule(
            UserRequest(duration=MATCH.d),
            TEST(lambda d: d < 10), # Automatically gets 'd'
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
            TEST(lambda d: d < 7), # Automatically gets 'd'
            TEST(lambda l1, l2, l3: l1 != l2 and l1 != l3 and l2 != l3) # Gets l1, l2, l3
        )
        def conflict_too_many_stops(self):
            if not any(isinstance(f, Warning) and "many stops" in f.get('message') for f in self.facts.values()):
                self.declare(Warning(message="Plan has many stops for a short trip. Consider focusing on one region."))

    # Return an *instance* of the engine
    return ItineraryEngine()



@contextmanager
def st_capture_stdout():
    """A context manager to capture stdout (print statements) and display them in Streamlit."""
    f = StringIO()
    with redirect_stdout(f):
        yield f




def run_expert_system(duration, month, interests):
    """
    This function now RUNS the engine and RETURNS the results.
    It no longer prints the final recommendations.
    """
    # Get the cached engine instance and reset it for a new run
    engine = get_knowledge_engine()
    engine.reset()
    
    engine.declare(UserRequest(duration=duration, month=month.lower(), interests=interests))

    # --- PASS 1: Detect Unknowns ---
    print("--- Experta PASS 1: Detecting unknown interests...")
    engine.run()
    print("--- Experta PASS 1: Complete.")

    tasks = [f for f in engine.facts.values() if isinstance(f, FindInfo)]
    
    if tasks:
        print(f"\n--- LLM Agent: Found {len(tasks)} items to research...")
        for task in tasks:
            interest = task.get('interest')
            # Call our cached LLM function
            llm_data = call_llm_agent(interest)
            if llm_data:
                engine.declare(Location(name=llm_data['name'],
                                        type=interest,
                                        region=llm_data['region']))
        print("--- LLM Agent: Research complete.\n")
    
    # --- PASS 2: Final Reasoning ---
    print("--- Experta PASS 2: Re-running engine with new knowledge...")
    engine.run()
    print("--- Experta PASS 2: Complete.")

    # --- Extract and Return Results ---
    warnings = [f.get('message') for f in engine.facts.values() if isinstance(f, Warning)]
    items = [f for f in engine.facts.values() if isinstance(f, ItineraryItem)]
    
    # Use set to return only unique warnings
    return list(set(warnings)), items



st.set_page_config(page_title="Sri Lankan Itinerary Advisor", page_icon="🇱🇰", layout="centered")

st.title("🇱🇰 Sri Lankan Itinerary Advisor")
st.markdown("A hybrid expert system using **`experta`** for logic and **Groq (Llama 3.1)** as a dynamic knowledge agent.")


st.header("1. Plan Your Trip", divider="blue")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    duration_input = st.number_input(
        "Trip Duration (days)", 
        min_value=1, 
        max_value=30, 
        value=7,
        help="How many days will you be traveling?"
    )

with col2:
    month_input = st.selectbox(
        "Month of Travel",
        [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ],
        index=7, # Default to August to test monsoon logic
        help="This is crucial for avoiding the monsoons!"
    )

# Use a text input for interests to allow for new ones
interests_input = st.text_input(
    "Your Interests (comma-separated)",
    value="beach, wildlife, surfing",
    help="Try: beach, history, hiking, wildlife, culture. Or test the LLM agent with 'surfing' or 'meditation'!"
)

# --- Button to run the system ---
if st.button("Generate Itinerary", type="primary", use_container_width=True):
    
    # Check if LLM is configured before running
    if not llm_model:
        st.error("Cannot generate itinerary. GROQ_API_KEY is not configured.")
    else:
        # Process inputs
        interests_list = [i.strip().lower() for i in interests_input.split(",") if i.strip()]
        
        st.header("2. Your Custom Itinerary", divider="blue")
        
        # --- Log Expander ---
        with st.expander("Show Processing Log (from Experta & LLM)"):
            log_placeholder = st.empty()
            
            with st_capture_stdout() as logs:
                # Run the main function
                warnings, recommendations = run_expert_system(duration_input, month_input, interests_list)
                log_output = logs.getvalue()
            
            # Display the captured logs
            log_placeholder.code(log_output)

        # --- Display Warnings ---
        if warnings:
            st.subheader("⚠️ Expert Warnings")
            for w in warnings:
                st.warning(w)

        # --- Display Recommendations ---
        st.subheader("🌴 Recommended Itinerary Items")
        if recommendations:
            # Tidy up the output
            unique_locations = {i.get('location'): i for i in recommendations}
            for i in unique_locations.values():
                st.success(f"**{i.get('location')}**: {i.get('reason')}")
        else:
            st.info("No itinerary items were generated. Try adjusting your interests or duration.")
