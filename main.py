from experta import *

#definition of custom fact types

class UserRequest(Fact):
    """
    Holds the user's request details.
    e.g., UserRequest(duration=7, month='august', interests=['beach', 'history'])
    """
    pass

class Location(Fact):
    """
    Holds the location details.
    e.g., Location(name='Sigiriya', type='history', region='cultural_triangle')
    """
    pass
    
class Weather(Fact):
    """
    Holds the weather details for a location.
    e.g., Weather(bad_region='south_west', month='august')
    """
    pass

class ItineraryItem(Fact):
    """
    A final recommendation for the user.
    e.g., ItineraryItem(location='Arugam Bay', reason='Good for beaches in August')
    """
    pass

class Warning(Fact):
    """
    A warning about a potential conflict or issue in the plan.
    e.g., Warning(message='Travel time between Sigiriya and Arugam Bay is high.')
    """
    pass

class Recommendation(Fact):
    """
    An intermediate fact used for reasoning.
    e.g., Recommendation(suggest_region='east_coast')
    """
    pass

#definition of the expert system knowledge engine

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
        yield Location(name='Arugam Bay', type='beach', region='east_coast')
        yield Location(name='Trincomalee', type='beach', region='east_coast')

        #weather data
        #south_west monsoon
        yield Weather(bad_region='south_west', month='june')
        yield Weather(bad_region='south_west', month='july')
        yield Weather(bad_region='south_west', month='august')

        #North_east monsoon
        yield Weather(bad_region='east_coast', month='december')
        yield Weather(bad_region='east_coast', month='january')
        yield Weather(bad_region='cultural_triangle', month='december')

        #rules for generating recommendations based on user requests and location/weather data

    @Rule(
        
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Weather(bad_region=MATCH.region, month=MATCH.month),
        TEST(lambda interests: 'beach' in interests),
        salience=100,
    )

    def determine_bad_weather_region(self, month, interests, region):
        """
        If the user wants beaches during a monsoon,
        flag the monsoon-affected region as "avoid".
        """

        self.declare(Recommendation(avoid_region=region))
        self.declare(Warning(message=f"Avoiding {region} for beaches due to monsoon in this month."))

    @Rule(
        UserRequest(month=MATCH.month, interests=MATCH.interests),
        Location(type='beach', region=MATCH.region),
        NOT(Recommendation(avoid_region=MATCH.region)),
        TEST(lambda interests: 'beach' in interests),
        salience=90
    )

    def determine_good_weather_region(self, region):
        """
        If the beach region is NOT in a bad weather zone,
        flag it in the "suggested" region for beaches.
        """
        
        if not any(isinstance(f, Recommendation) and f.get('suggest_region') == region for f in self.facts.values()):
            self.declare(Recommendation(suggest_region=region))

    @Rule(
        UserRequest(interests=MATCH.interests),
        Location(name=MATCH.name, type='history'),
        TEST(lambda interests: 'history' in interests)
    )

    def suggest_history(self, name):
        self.declare(ItineraryItem(location=name, reason="Matches 'history' interest"))


    @Rule(
        UserRequest(interests=MATCH.interests),
        Location(name=MATCH.name, type='hiking'),
        TEST(lambda interests: 'hiking' in interests)
    )
    def suggest_hiking(self, name):
        self.declare(ItineraryItem(location=name, reason="Matches 'hiking' interest"))

    @Rule(
        UserRequest(interests=MATCH.interests),
        Location(name=MATCH.name, type='culture'),
        TEST(lambda interests: 'culture' in interests)
    )
    def suggest_culture(self, name):
        self.declare(ItineraryItem(location=name, reason="Matches 'culture' interest"))
    
    @Rule(
        UserRequest(interests=MATCH.interests),
        Location(name=MATCH.name, type='wildlife', region=MATCH.region),
        NOT(Recommendation(avoid_region=MATCH.region)),
        TEST(lambda interests: 'wildlife' in interests)
    )
    def suggest_wildlife(self, name):
        self.declare(ItineraryItem(location=name, reason="Matches 'wildlife' interest"))

    @Rule(
        
        Recommendation(suggest_region=MATCH.region),
        Location(name=MATCH.name, type='beach', region=MATCH.region)
    )
    def suggest_good_beach(self, name, region):
        """
        Suggests beaches ONLY in the "good weather" region
        identified in Phase 1.
        """
        self.declare(ItineraryItem(location=name, reason=f"Good beach in {region} this month"))

    @Rule(
        UserRequest(duration=MATCH.d),
        TEST(lambda d: d < 10), # Only for short trips
        # Check if BOTH these facts have been declared by previous rules
        ItineraryItem(location='Sigiriya'), 
        ItineraryItem(location='Arugam Bay')
    )
    def conflict_travel_time_sigiriya_arugam(self):
        """
        Warns if the plan includes two locations that are
        far apart, on a short trip.
        """
        self.declare(Warning(message="High travel time between Cultural Triangle (Sigiriya) and East Coast (Arugam Bay). Difficult in < 10 days."))

    @Rule(
        UserRequest(duration=MATCH.d),
        TEST(lambda d: d < 7), # Only for very short trips
        ItineraryItem(location=MATCH.l1),
        ItineraryItem(location=MATCH.l2),
        ItineraryItem(location=MATCH.l3),
        # Check for 3 distinct locations
        TEST(lambda l1, l2, l3: l1 != l2 and l1 != l3 and l2 != l3) 
    )
    def conflict_too_many_stops(self):
        """
        Warns if a very short trip has too many suggested items.
        (This rule will fire multiple times, a more complex implementation
        would use a counter, but this demonstrates the concept)
        """
        if not any(isinstance(f, Warning) and "many stops" in f.get('message') for f in self.facts.values()):
            self.declare(Warning(message="Plan has many stops for a short trip. Consider focusing on one region."))

def get_itinerary(duration, month, interests):
    """
    A helper function to run the expert system.
    """

    engine = ItineraryEngine()
    engine.reset()

    engine.declare(UserRequest(duration=duration, month=month.lower(), interests=interests))

    engine.run()

    print("---" * 10)
    print("Generating Iterary Recommendations for:")
    print(f"   Duration: {duration} days")
    print(f"   Month: {month.capitalize()}")
    print(f"   Interests: {', '.join(interests)}")
    print("---" * 10)

    print("\n Warnings:")
    warnings = [f.get('message') for f in engine.facts.values() if isinstance(f, Warning)]
    if warnings:
        for w in set(warnings):
            print(f"  - {w}")
    else:
        print("  - No conflicts found. Plan looks good!")

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
        month='August',
        interests=['beach', 'wildlife']
    )

    get_itinerary(
        duration=10,
        month="January",
        interests=['beach', 'hiking', 'culture']
    )

    get_itinerary(
        duration=5,
        month="July",
        interests=['beach', 'history']
    )
