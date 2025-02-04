import sqlite3
import json
import csv
import os
from datetime import datetime 
from collections import Counter


# Connect to SQLite database
db_path = 'game_04_database.db'
print(f"Opening database at: {os.path.abspath(db_path)}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables (with JSON data stored as TEXT)
cursor.execute('''
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS battery_log (
    log_id INTEGER PRIMARY KEY,
    battery_constitution TEXT,
    player_id INTEGER,
    game_hour INTEGER,
    timestamp DATETIME,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS game_hour_data (
    game_hour_id INTEGER PRIMARY KEY,
    player_id INTEGER,
    game_hour INTEGER CHECK(game_hour BETWEEN 6 AND 18),
    consumption INTEGER,
    production INTEGER,
    game_hour_constitution TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
''')

conn.commit()


def init_player(name):
    cursor.execute("INSERT INTO players (name) VALUES (?)", (name,))
    conn.commit()
    
    
def init_players_csv(csv_file_path):
    
    csv_path =  os.path.abspath(csv_file_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")

    """Load production and consumption data from the CSV file into the database."""
    #Open CSV file containing initialised player values,load them into a database
    with open(csv_path, "r", newline="" ) as file:
        reader = csv.reader(file)
        print("File opened successfully")
        headers = next(reader)
        for row in reader:
            player_name = row[headers.index('PlayerName')]
            hour = int(row[headers.index('Hour')])
            consumption = int(row[headers.index('Consumption')])
            production = int(row[headers.index('Production')])
            
            # Get the player_id for the player name
            cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
            player_id = cursor.fetchone()
            
            if player_id:
                player_id = player_id[0]
                print(f"Player {player_name} found. Inserting data for hour {hour}")
            else:
                print(f"Player {player_name} not found.")
                continue

            # Insert the data into the game_hours table
            cursor.execute('''
                INSERT OR REPLACE INTO game_hour_data (player_id, game_hour, consumption, production)
                VALUES (?, ?, ?, ?)
            ''', (player_id, hour, consumption, production))
            
    # Commit the changes
    conn.commit()
    print("CSV file processing complete.")
    
    # Function to read battery constitution from a CSV and insert JSON into the database
def init_battery_logs(csv_filepath):
    with open(csv_filepath, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
            battery_constitution_json = json.dumps(row)  # Convert list to JSON string
            player_id = 99  # Generate random player_id
            game_hour = 0  # Random hour between 0-23
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp
            
            # Insert into database
            cursor.execute('''
                INSERT INTO battery_log (battery_constitution, player_id, game_hour, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (battery_constitution_json, player_id, game_hour, timestamp))

    conn.commit()
    print("Battery log entries inserted successfully!")
    


def update_player_game_hour_constitution(player_name, game_hour, difference):
    """Update the player's game hour constitution based on the difference."""
    
    # Get the player_id from the players table
    cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
    player_id = cursor.fetchone()
    
    if not player_id:
        print(f"Error: Player '{player_name}' not found.")
        return
    
    player_id = player_id[0]  # Extract the player_id from the tuple

    # Fetch the current game_hour_constitution, production, and consumption for the player and game_hour
    cursor.execute(''' 
    SELECT game_hour_constitution, production, consumption 
    FROM game_hour_data 
    WHERE player_id = ? AND game_hour = ? 
    ''', (player_id, game_hour))
    
    result = cursor.fetchone()

    if result:
        # Extract current constitution, production, and consumption values
        current_constitution = json.loads(result[0])  # Convert JSON string to list
        production = result[1]  # Extract production value
        consumption = result[2]  # Extract consumption value
    else:
        # If no entry exists, assume empty constitution and default production/consumption
        current_constitution = []
        production = 0  # Default production value if no record exists
        consumption = 0  # Default consumption value if no record exists

    # Call the function to calculate the new constitution
    new_constitution = calculate_new_player_hour_constitution(current_constitution, difference, production, consumption)

    # Convert the updated constitution list back to a JSON string for database storage
    new_constitution_json = json.dumps(new_constitution)

    # Update the game_hours table with the new constitution
    cursor.execute(''' 
    INSERT INTO game_hour_data (player_id, game_hour, game_hour_constitution)
    VALUES (?, ?, ?)
    ON CONFLICT(player_id, game_hour)
    DO UPDATE SET game_hour_constitution = excluded.game_hour_constitution
    ''', (player_id, game_hour, new_constitution_json))
    
    conn.commit()



def calculate_new_player_hour_constitution(current_constitution, difference, production, consumption):
    """
    Calculate the new constitution based on the current constitution, difference, production, and consumption.
    """
    
    # Step 1: Calculate the total difference based on the difference list
    total_difference = 0  # To track the sum of all the differences

    # Apply the difference list: Add or remove according to the difference sign
    for diff in difference:
        if diff.startswith('+'):
            total_difference -= 1  # A positive difference means removing one
        elif diff.startswith('-'):
            total_difference += 1  # A negative difference means adding one

    # Step 2: Apply the difference to the current constitution
    updated_constitution = current_constitution.copy()

    # Remove items from the current constitution based on the '+X' in the difference
    for diff in difference:
        if diff.startswith('+'):
            letter_to_remove = diff[1]
            if letter_to_remove in updated_constitution:
                updated_constitution.remove(letter_to_remove)

    # Add items to the constitution based on the '-X' in the difference
    for diff in difference:
        if diff.startswith('-'):
            letter_to_add = diff[1]
            updated_constitution.append(letter_to_add)

    # Step 3: Calculate the "Grid" value (consumption adjustment)
    grid_value = total_difference - consumption

    # Step 4: Add 'G' based on the grid value
    if grid_value < 0:
        # If the grid is negative, add 'G' for each negative value
        updated_constitution.extend(['G'] * abs(grid_value))

    # Step 5: Return the new constitution list
    return updated_constitution


def log_battery_data_and_update_constitution(player_name, game_hour, battery_constitution):
    """Log battery data and update the player's game hour constitution."""
    #First use the data in string format to calculate change in battery constitution
    difference = compare_battery_log(battery_constitution)
    
    # Convert battery_constitution list to a JSON string
    battery_constitution_json = json.dumps(battery_constitution)
    
    # Get player_id based on player name
    cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
    player_id = cursor.fetchone()[0]
    
    # Insert battery log data
    cursor.execute('''
    INSERT INTO battery_log (battery_constitution, player_id, game_hour, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (battery_constitution_json, player_id, game_hour, datetime.now()))
    conn.commit()
    
    # If a previous log exists, update constitution
    if difference is not None:
        update_player_game_hour_constitution(player_name, game_hour, difference)
    
def compare_battery_log(new_battery_constitution):
    """Compare the new battery constitution with the most recent logged one (handling letters)."""
    
    # Retrieve the last logged battery constitution (from any player)
    cursor.execute('''
    SELECT battery_constitution FROM battery_log
    ORDER BY timestamp DESC
    LIMIT 1
    ''')
    
    last_log = cursor.fetchone()
    
    if not last_log:
        return None  # No previous logs exist
    
    # Convert JSON string back to a Python list
    last_battery_constitution = json.loads(last_log[0])

    # Count occurrences of each letter
    old_counts = Counter(last_battery_constitution)
    new_counts = Counter(new_battery_constitution)

    # Compute differences
    difference = []

    # Check for removed elements (-X)
    for letter, old_count in old_counts.items():
        new_count = new_counts.get(letter, 0)  # Default to 0 if letter isn't in new list
        if old_count > new_count:
            for _ in range(old_count - new_count):
                difference.append(f'-{letter}')

    # Check for added elements (+X)
    for letter, new_count in new_counts.items():
        old_count = old_counts.get(letter, 0)  # Default to 0 if letter wasn't in old list
        if new_count > old_count:
            for _ in range(new_count - old_count):
                difference.append(f'+{letter}')

    return difference


def print_battery_log():
    cursor.execute('SELECT * FROM battery_log')
    rows = cursor.fetchall()
    for row in rows:
        log_id, battery_constitution, player_id, game_hour, timestamp = row
        # Convert JSON string back to list
        battery_constitution_list = json.loads(battery_constitution)
        print(f"Log ID: {log_id}, Player ID: {player_id}, Game Hour: {game_hour}, Timestamp: {timestamp}")
        print(f"Battery Constitution: {battery_constitution_list}")
        print("------------")
        
def print_player_game_hours(player_name):
    cursor.execute('''
    SELECT g.game_hour, g.consumption, g.production, g.game_hour_constitution
    FROM game_hour_data g
    JOIN players p ON g.player_id = p.player_id
    WHERE p.name = ?
    ORDER BY g.game_hour
    ''', (player_name,))
    
    rows = cursor.fetchall()
    
    for row in rows:
        game_hour, consumption, production, game_hour_constitution = row
        # Convert JSON string back to list
        game_hour_constitution_list = json.loads(game_hour_constitution)
        print(f"Game Hour: {game_hour}, Consumption: {consumption}, Production: {production}")
        print(f"Game Hour Constitution: {game_hour_constitution_list}")
        print("------------")


# Add players: Teal, Orange, Magenta, Yellow
for player_name in ['Teal', 'Orange', 'Magenta', 'Yellow']:
    init_player(player_name)
    
init_players_csv("SQLite_Testing\sqlite_prosum.csv")
#init_battery_logs("SQLite_Testing\sqlite_battery.csv")













