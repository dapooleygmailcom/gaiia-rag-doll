import os
import csv
import random

OUTPUT_DIR = "data/analysis"
FILE_PATH = os.path.join(OUTPUT_DIR, "airbnb_listings.csv")

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]
NEIGHBORHOODS = ["Manly", "Bondi", "Surry Hills", "Newtown", "Darlinghurst", "Parramatta"]
DESCRIPTIONS = [
    "A lovely, quiet neighborhood perfect for families. Close to the beach.",
    "Very noisy and bustling, great for nightlife but hard to sleep.",
    "Extremely clean and modern apartment with high-speed wifi.",
    "A bit outdated but very affordable and cozy.",
    "Luxury villa with a pool and ocean views. Incredible experience.",
    "Basic room, clean but tiny. The host was very friendly.",
    "Terrible experience, the place was dirty and loud.",
    "Quiet and peaceful, the perfect romantic getaway."
]

def generate_dataset(num_rows=5000):
    print("Generating synthetic Airbnb dataset...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(FILE_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["id", "neighborhood", "room_type", "price", "rating", "review_text"])
        
        for i in range(1, num_rows + 1):
            neighborhood = random.choice(NEIGHBORHOODS)
            room_type = random.choice(ROOM_TYPES)
            price = round(random.uniform(50.0, 800.0), 2)
            rating = round(random.uniform(2.5, 5.0), 1)
            review = random.choice(DESCRIPTIONS)
            
            writer.writerow([i, neighborhood, room_type, price, rating, review])
            
    print(f"Generated {num_rows} rows at {FILE_PATH}")

if __name__ == "__main__":
    generate_dataset()
