import sys
import os
sys.path.append('.')
from data import substation

# Test the substation class
print("Testing substation class...")
print("=" * 50)

# Create a substation object
try:
    sub_a = substation('A')
    
    print(f"Substation ID: {sub_a.get_substation_id()}")
    print(f"Substation Name: {sub_a.get_substation_name()}")
    print(f"Coordinates: {sub_a.get_coordinates()}")
    print(f"Number of feeders: {sub_a.get_feeder_count()}")
    print(f"Feeder IDs: {sub_a.get_feeder_ids()}")
    
    print("\nFeeder details:")
    for feeder in sub_a.get_feeders():
        print(f"  - Feeder {feeder.get_feeder_id()}: {len(feeder.get_consumption_data()) if feeder.get_consumption_data() is not None else 0} data points")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()