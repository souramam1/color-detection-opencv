import sqlite3
import json
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('game_data.db')
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
CREATE TABLE IF NOT EXISTS game_hours (
    game_hour_id INTEGER PRIMARY KEY,
    player_id INTEGER,
    game_hour INTEGER CHECK(game_hour BETWEEN 1 AND 12),
    consumption INTEGER,
    production INTEGER,
    game_hour_constitution TEXT,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
''')

conn.commit()


def add_player(name):
    cursor.execute("INSERT INTO players (name) VALUES (?)", (name,))
    conn.commit()
    
    
def log_game_hour_data(player_name, game_hour, consumption, production, game_hour_constitution):
    # Convert the game_hour_constitution list to a JSON string
    game_hour_constitution_json = json.dumps(game_hour_constitution)
    
    # Get player_id based on player name
    cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
    player_id = cursor.fetchone()[0]
    
    # Insert the data into the game_hours table
    cursor.execute('''
    INSERT INTO game_hours (player_id, game_hour, consumption, production, game_hour_constitution)
    VALUES (?, ?, ?, ?, ?)
    ''', (player_id, game_hour, consumption, production, game_hour_constitution_json))
    conn.commit()
    
def update_player_game_hour_constitution(player_name, game_hour, new_game_hour_constitution):
    """Update the game hour constitution for a player at a specific game hour."""
    # Convert the new game_hour_constitution list to a JSON string
    new_game_hour_constitution_json = json.dumps(new_game_hour_constitution)
    
    # Get player_id based on player name
    cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
    player_id = cursor.fetchone()
    
    if player_id:
        player_id = player_id[0]
        
        # Update the game_hour_constitution for the specified player and game hour
        cursor.execute('''
        UPDATE game_hours
        SET game_hour_constitution = ?
        WHERE player_id = ? AND game_hour = ?
        ''', (new_game_hour_constitution_json, player_id, game_hour))
        
        # Commit the changes to the database
        conn.commit()
        
        print(f"Updated game hour constitution for {player_name} at hour {game_hour}.")
    else:
        print(f"Player {player_name} not found.")

def calculate_player_hour_constitution():

def log_battery_data_and_update_constitution(player_name, game_hour, battery_constitution):
    """Log battery data and update the player's game hour constitution."""
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
    
    # Now update the player's game hour constitution
    update_player_game_hour_constitution(player_name, game_hour)


def update_player_game_hour_constitution(player_name, game_hour):
    """Update the player's game hour constitution based on battery log and game hour data."""
    # Get the player data (consumption, production, etc.)
    cursor.execute('''
    SELECT g.consumption, g.production, g.game_hour_constitution 
    FROM game_hours g
    JOIN players p ON g.player_id = p.player_id
    WHERE p.name = ? AND g.game_hour = ?
    ''', (player_name, game_hour))
    
    player_data = cursor.fetchone()
    if not player_data:
        print(f"No data found for player {player_name} at game hour {game_hour}.")
        return
    
    consumption, production, _ = player_data  # Ignore existing constitution for now
    
    # Calculate the player's constitution for that game hour
    cursor.execute('''
    SELECT battery_constitution FROM battery_log
    WHERE player_id = (SELECT player_id FROM players WHERE name = ?)
    AND game_hour = ?
    ORDER BY timestamp DESC
    LIMIT 1
    ''', (player_name, game_hour))
    
    battery_constitution_json = cursor.fetchone()[0]
    battery_constitution = json.loads(battery_constitution_json)
    
    # Step 1: If production >= consumption, just use player's initial for constitution
    if production >= consumption:
        game_hour_constitution = [player_name[0]] * consumption
    else:
        # Step 2: If production < consumption, calculate deficit
        deficit = consumption - production
        battery_tokens_used = 0
        
        # Step 3: Try to use tokens from the battery
        if battery_constitution:
            tokens_from_battery = min(deficit, len(battery_constitution))
            battery_tokens_used = tokens_from_battery
            game_hour_constitution = battery_constitution[:tokens_from_battery]
            deficit -= tokens_from_battery
        
        # Step 4: Use tokens from the grid if deficit remains
        game_hour_constitution.extend(['G'] * deficit)
    
    # Step 5: Convert game hour constitution list back to JSON
    game_hour_constitution_json = json.dumps(game_hour_constitution)
    
    # Step 6: Update the player's game hour constitution in the database
    cursor.execute('''
    UPDATE game_hours
    SET game_hour_constitution = ?
    WHERE player_id = (SELECT player_id FROM players WHERE name = ?)
    AND game_hour = ?
    ''', (game_hour_constitution_json, player_name, game_hour))
    
    # Commit changes to the database
    conn.commit()
    print(f"Updated game hour constitution for {player_name} at hour {game_hour}.")


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
    FROM game_hours g
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

# Example: Log data for Player Teal at Game Hour 1
game_hour_constitution_example = [
    "Resource A present",
    "Energy level high",
    "Solar panel functioning",
    "Battery charging"
]

# Example: Log battery data for Player Teal at Game Hour 1
battery_constitution_example = [
    100,  # Battery capacity
    10,   # Discharge rate
    50,   # Solar charge contribution
    20    # Battery level
]

# Add players: Teal, Orange, Magenta, Yellow
for player_name in ['Teal', 'Orange', 'Magenta', 'Yellow']:
    add_player(player_name)
    

log_game_hour_data("Teal", 1, 50, 30, game_hour_constitution_example)
log_battery_data("Teal", 1, battery_constitution_example)
print_battery_log()
print_player_game_hours("Teal")









