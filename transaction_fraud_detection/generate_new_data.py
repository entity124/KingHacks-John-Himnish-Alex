import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
import uuid

# Configuration
NUM_USERS = 300
# Shifted dates forward by ~6 months
START_DATE = datetime(2023, 7, 1) 
END_DATE = datetime(2024, 12, 30)
OUTPUT_FILE = "mock_transactions-2.csv"
TRAIN_OUTPUT_FILE = "train-2.jsonl"

# Modified Merchants (New Names) and slightly tweaked amount ranges
NORMAL_MERCHANTS = [
    ("FreshFoods", "Groceries", 55, 210, ["Groceries", "Weekly Shop", "Store #905"]), # Was GroceryMart
    ("JavaJoint", "Dining", 5, 18, ["Coffee", "Latte", "Morning Brew"]),             # Was CoffeeSpot
    ("CityRail", "Transport", 2.50, 12, ["Bus Fare", "Subway Token", "Ride Share"]), # Was MetroTransit
    ("GadgetDepot", "Electronics", 25, 520, ["Gadget", "Cable", "Repair"]),          # Was TechStore
    ("BingeWatch", "Entertainment", 13, 14, ["Monthly Sub", "Streaming Service"]),   # Was StreamFlix
    ("FitPhysique", "Health", 45, 45, ["Gym Membership", "Monthly Dues"]),           # Was GymBody
    ("UrbanElectric", "Utilities", 85, 160, ["Electric Bill", "Power Usage"]),       # Was CityPower
    ("CellConnect", "Utilities", 48, 65, ["Phone Bill", "Data Plan"]),               # Was MobileNet
    ("QuickBite", "Dining", 12, 35, ["Lunch", "Burger Combo", "Drive Thru"]),        # Was FastBurger
    ("ReadAddict", "Shopping", 18, 65, ["Books", "Novel", "Stationery"]),            # Was BookWorm
    ("TrendSetter", "Shopping", 35, 160, ["Clothing", "Shoes", "Accessory"]),        # Was FashionHub
    ("TravelEasy", "Transport", 18, 50, ["Ride to Work", "Trip", "Travel Svc"]),     # Was RideGo
]

def generate_random_date(start, end):
    delta = end - start
    random_days = random.randrange(delta.days)
    random_seconds = random.randrange(86400)
    return start + timedelta(days=random_days, seconds=random_seconds)

def generate_user_data(user_id):
    transactions = []
    
    # 1. Generate Normal Background Noise
    num_txns = random.randint(40, 120)
    for _ in range(num_txns):
        merchant, category, min_amt, max_amt, descs = random.choice(NORMAL_MERCHANTS)
        # Apply a slight random multiplier to vary it from the original dataset's logic
        amount = round(random.uniform(min_amt, max_amt), 2)
        if min_amt == max_amt: 
            amount = min_amt
            
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": generate_random_date(START_DATE, END_DATE),
            "user_id": user_id,
            "merchant": merchant,
            "amount": amount,
            "description": random.choice(descs),
            "category": category,
            "pattern_label": "normal"
        })

    # 2. Inject Predatory Patterns
    # We allow multiple potential issues per user
    
    # Scenario A: Price Jump (Recurring sub doubles)
    if random.random() < 0.15:
        base_date = START_DATE + timedelta(days=random.randint(0, 60))
        merch = "StealthNet" # Was SneakyVPN
        # Stable phase
        for i in range(4):
            transactions.append({
                "transaction_id": str(uuid.uuid4()),
                "timestamp": base_date + timedelta(days=30*i),
                "user_id": user_id,
                "merchant": merch,
                "amount": 10.99, # Slightly adjusted from 9.99
                "description": "Monthly Access",
                "category": "Software",
                "pattern_label": "normal"
            })
        # The Jump
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date + timedelta(days=120),
            "user_id": user_id,
            "merchant": merch,
            "amount": 54.99, # Adjusted from 49.99
            "description": "Monthly Access Premium",
            "category": "Software",
            "pattern_label": "predatory_jump"
        })

    # Scenario B: Inactivity Fee (Dormant then charged)
    if random.random() < 0.15:
        base_date = START_DATE + timedelta(days=random.randint(0, 60))
        merch = "AncientTrust" # Was OldBank
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date,
            "user_id": user_id,
            "merchant": merch,
            "amount": 210.00,
            "description": "Opening Deposit",
            "category": "Finance",
            "pattern_label": "normal"
        })
        # Fee after 5 months silence
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date + timedelta(days=150),
            "user_id": user_id,
            "merchant": merch,
            "amount": 28.50, # Adjusted from 25.00
            "description": "Dormancy Fee",
            "category": "Finance",
            "pattern_label": "predatory_inactivity"
        })

    # Scenario C: Hidden Subscription (Trap)
    if random.random() < 0.15:
        base_date = START_DATE + timedelta(days=random.randint(0, 150))
        merch = "BonusTech" # Was FreeGadget
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date,
            "user_id": user_id,
            "merchant": merch,
            "amount": 5.95, # Adjusted from 4.95
            "description": "S&H for Trial",
            "category": "Shopping",
            "pattern_label": "potential_trap"
        })
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date + timedelta(days=14),
            "user_id": user_id,
            "merchant": merch,
            "amount": 105.00, # Adjusted from 99.00
            "description": "Quarterly Membership",
            "category": "Shopping",
            "pattern_label": "predatory_hidden"
        })

    # Scenario D: Creeping Fee (Slow boil)
    if random.random() < 0.10:
        base_date = START_DATE + timedelta(days=random.randint(0, 30))
        merch = "ViewMax" # Was StreamPlus
        base_amt = 12.50 # Adjusted base
        for i in range(6):
            amt = round(base_amt * (1.08 ** i), 2) # 8% increase each time
            transactions.append({
                "transaction_id": str(uuid.uuid4()),
                "timestamp": base_date + timedelta(days=30*i),
                "user_id": user_id,
                "merchant": merch,
                "amount": amt,
                "description": "Streaming Plan",
                "category": "Entertainment",
                "pattern_label": "predatory_creeping" if i > 3 else "normal"
            })

    # Scenario E: Double Billing (Accidental duplicate)
    if random.random() < 0.10:
        base_date = generate_random_date(START_DATE, END_DATE)
        merch = "BuggyShop" # Was GlitchyStore
        amt = 48.25 # Adjusted from 45.50
        # Charge 1
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date,
            "user_id": user_id,
            "merchant": merch,
            "amount": amt,
            "description": "Purchase #9921",
            "category": "Shopping",
            "pattern_label": "normal"
        })
        # Charge 2 (Same day, slightly different time)
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date + timedelta(seconds=random.randint(10, 300)),
            "user_id": user_id,
            "merchant": merch,
            "amount": amt,
            "description": "Purchase #9921",
            "category": "Shopping",
            "pattern_label": "predatory_double"
        })

    # Scenario F: Zombie Subscription (Returns from dead)
    if random.random() < 0.10:
        base_date = START_DATE
        merch = "UndeadFitness" # Was ZombieGym
        # Users cancels
        for i in range(3):
            transactions.append({
                "transaction_id": str(uuid.uuid4()),
                "timestamp": base_date + timedelta(days=30*i),
                "user_id": user_id,
                "merchant": merch,
                "amount": 32.99, # Adjusted from 29.99
                "description": "Monthly Gym",
                "category": "Health",
                "pattern_label": "normal"
            })
        # 6 Month Gap
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": base_date + timedelta(days=240), # 8 months later
            "user_id": user_id,
            "merchant": merch,
            "amount": 32.99,
            "description": "Monthly Gym",
            "category": "Health",
            "pattern_label": "predatory_zombie"
        })

    return transactions

def main():
    print(f"Generating advanced data for {NUM_USERS} users...")
    all_transactions = []
    
    # Generate completely new random User IDs to differentiate from file 1
    # Use a set to ensure uniqueness
    user_ids = set()
    while len(user_ids) < NUM_USERS:
        user_ids.add(f"user_{random.randint(1000, 9999)}")
    
    for user_id in list(user_ids):
        all_transactions.extend(generate_user_data(user_id))
        
    df = pd.DataFrame(all_transactions)
    df = df.sort_values("timestamp")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} transactions to {OUTPUT_FILE}")
    
    # Generate Training Data (JSONL)
    print(f"Generating training dataset ({TRAIN_OUTPUT_FILE})...")
    training_data = []
    
    grouped = df.groupby(['user_id', 'merchant'])
    for (user, merchant), group in grouped:
        if len(group) < 2: continue
        
        sorted_txns = group.sort_values("timestamp").to_dict('records')
        clean_txns = []
        for t in sorted_txns:
            clean_txns.append({
                "date": str(t.get("timestamp"))[:10],
                "amt": t.get("amount"),
                "desc": t.get("description")
            })
            
        predatory_txns = group[group['pattern_label'].str.startswith('predatory')]
        
        is_predatory = len(predatory_txns) > 0
        pattern_type = "None"
        reason = "Transactions show normal consistent activity."
        
        if is_predatory:
            labels = predatory_txns['pattern_label'].unique()
            if "predatory_jump" in labels:
                pattern_type = "Price Jump"
                reason = "Subscription price increased >50%."
            elif "predatory_inactivity" in labels:
                pattern_type = "Inactivity Fee"
                reason = "Fee charged after long period of inactivity."
            elif "predatory_hidden" in labels:
                pattern_type = "Hidden Subscription"
                reason = "Trial converted to high recurring fee."
            elif "predatory_creeping" in labels:
                pattern_type = "Creeping Fee"
                reason = "Subscription cost increased multiple times sequentially."
            elif "predatory_double" in labels:
                pattern_type = "Double Billing"
                reason = "Duplicate charge for same amount on same day."
            elif "predatory_zombie" in labels:
                pattern_type = "Zombie Subscription"
                reason = "Recurring charge resumed after >6 months of silence."

        sample = {
            "instruction": "Analyze the transaction history for predatory patterns. Return JSON: {is_predatory, pattern_type, reason}.",
            "input": json.dumps(clean_txns),
            "output": json.dumps({
                "is_predatory": is_predatory,
                "pattern_type": pattern_type,
                "reason": reason
            })
        }
        training_data.append(sample)
        
    with open(TRAIN_OUTPUT_FILE, "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(training_data)} training examples to {TRAIN_OUTPUT_FILE}")

if __name__ == "__main__":
    main()