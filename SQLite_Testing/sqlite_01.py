import sqlite3
import json
from datetime import datetime

class GameDatabase:
    def __init__(self, db_name="game.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        """Creates the necessary tables if they do not already exist."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS battery_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_hour INTEGER NOT NULL CHECK(game_hour BETWEEN 6 AND 18),
            player_id INTEGER NOT NULL,
            battery_slots TEXT NOT NULL,  
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            game_hour INTEGER NOT NULL CHECK(game_hour BETWEEN 6 AND 18),
            action TEXT NOT NULL,  
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
        """)

        # New table to store player's consumption, production, and constitution
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            game_hour INTEGER NOT NULL CHECK(game_hour BETWEEN 6 AND 18),
            consumption INTEGER NOT NULL,
            production INTEGER NOT NULL,
            constitution TEXT NOT NULL,
            FOREIGN KEY (player_id) REFERENCES players(id),
            UNIQUE(player_id, game_hour)  -- Ensures one entry per player per hour
        )
        """)

        self.connection.commit()

    def add_player(self, name):
        """Adds a new player to the database if they don't already exist."""
        try:
            self.cursor.execute("INSERT OR IGNORE INTO players (name) VALUES (?)", (name,))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error adding player: {e}")

    def get_player_id(self, name):
        """Retrieves a player's ID based on their name."""
        self.cursor.execute("SELECT id FROM players WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_previous_battery_state(self, game_hour):
        """Fetches the last recorded battery state for a given game hour."""
        self.cursor.execute("""
        SELECT battery_slots FROM battery_log WHERE game_hour = ? 
        ORDER BY timestamp DESC LIMIT 1
        """, (game_hour,))
        result = self.cursor.fetchone()
        return json.loads(result[0]) if result else None

    def log_battery_state(self, player_name, game_hour, battery_state, consumption, production):
        """Logs the battery state after a player's move and determines their action."""
        player_id = self.get_player_id(player_name)
        if player_id is None:
            print(f"Player {player_name} not found.")
            return

        # Convert battery_state to JSON
        battery_json = json.dumps(battery_state)

        # Get previous battery state
        previous_state = self.get_previous_battery_state(game_hour)

        # Insert new battery state log
        self.cursor.execute("""
        INSERT INTO battery_log (game_hour, player_id, battery_slots)
        VALUES (?, ?, ?)
        """, (game_hour, player_id, battery_json))
        self.connection.commit()

        # Determine the player's action
        if previous_state:
            action = self.determine_player_action(previous_state, battery_state)
            if action:
                self.log_player_move(player_id, game_hour, action)

        # Log player data (consumption, production, constitution)
        constitution_json = json.dumps(battery_state)  # Save battery state as constitution
        self.cursor.execute("""
        INSERT OR REPLACE INTO player_data (player_id, game_hour, consumption, production, constitution)
        VALUES (?, ?, ?, ?, ?)
        """, (player_id, game_hour, consumption, production, constitution_json))
        self.connection.commit()

    def determine_player_action(self, previous_state, new_state):
        """Compares two battery states and determines the player's action."""
        actions = []
        for i in range(16):  # 16 battery slots
            if previous_state[i] != new_state[i]:
                if new_state[i] == "empty":
                    actions.append(f"removed chip from slot {i + 1}")
                else:
                    actions.append(f"added chip to slot {i + 1} ({new_state[i]})")

        return ", ".join(actions) if actions else None

    def log_player_move(self, player_id, game_hour, action):
        """Logs the player's action for the given game hour."""
        self.cursor.execute("""
        INSERT INTO player_moves (player_id, game_hour, action)
        VALUES (?, ?, ?)
        """, (player_id, game_hour, action))
        self.connection.commit()

    def get_player_moves(self, player_name):
        """Retrieves all recorded moves for a player."""
        player_id = self.get_player_id(player_name)
        if player_id is None:
            print(f"Player {player_name} not found.")
            return

        self.cursor.execute("""
        SELECT game_hour, action, timestamp FROM player_moves
        WHERE player_id = ? ORDER BY game_hour
        """, (player_id,))

        moves = self.cursor.fetchall()
        for move in moves:
            game_hour, action, timestamp = move
            print(f"At {game_hour}: {player_name} {action} (Recorded at {timestamp})")

    def get_player_data(self, player_name):
        """Retrieves the consumption, production, and constitution data for each game hour."""
        player_id = self.get_player_id(player_name)
        if player_id is None:
            print(f"Player {player_name} not found.")
            return

        self.cursor.execute("""
        SELECT game_hour, consumption, production, constitution FROM player_data
        WHERE player_id = ? ORDER BY game_hour
        """, (player_id,))

        data = self.cursor.fetchall()
        for record in data:
            game_hour, consumption, production, constitution_json = record
            constitution = json.loads(constitution_json)
            print(f"At {game_hour}: Consumption={consumption}, Production={production}, Constitution={constitution}")

    def close(self):
        """Closes the database connection."""
        self.connection.close()

# -----------------------------------------------
# Example Usage
# -----------------------------------------------

# Initialize the database
db = GameDatabase()

# Add players
for player in ["Magenta", "Teal", "Orange", "Yellow"]:
    db.add_player(player)

# Example game moves with consumption and production values
battery_state_6am = ["A", "B", "C", "empty", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
db.log_battery_state("Magenta", 6, battery_state_6am, consumption=10, production=5)
db.log_battery_state("Teal", 6, battery_state_6am, consumption=8, production=7)

# Retrieve and display player data (consumption, production, constitution)
db.get_player_data("Magenta")
db.get_player_data("Teal")

# Retrieve and display player moves
db.get_player_moves("Magenta")
db.get_player_moves("Teal")

# Close database
db.close()
