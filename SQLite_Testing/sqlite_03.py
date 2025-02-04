import sqlite3
import json
import csv
import os
from datetime import datetime
from collections import Counter


class GameDatabase:
    def __init__(self, db_path=f'game_06_database.db', players=['Teal','Orange','Magenta','Yellow']):
        """Initialize the database connection and create tables."""
        self.db_path = db_path
        print(f"Opening database at: {os.path.abspath(self.db_path)}")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()
        for player in players:
            self.insert_player(player)
        self.init_player_data_from_csv("SQLite_Testing\sqlite_prosum.csv")
        self.init_battery_data_from_csv("SQLite_Testing\sqlite_battery.csv")
        

    ################ INITIALISATION METHODS ###################
    def create_tables(self):
        """Creates all necessary tables in the database."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE
        );
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS battery_log (
            log_id INTEGER PRIMARY KEY,
            battery_constitution TEXT,
            player_id INTEGER,
            game_hour INTEGER,
            timestamp DATETIME,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );
        ''')

        self.cursor.execute('''
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

        self.conn.commit()

    def close_connection(self):
        """Closes the database connection."""
        self.conn.close()

    def insert_player(self, name):
        """Inserts a new player into the database."""
        self.cursor.execute("INSERT OR IGNORE INTO players (name) VALUES (?)", (name,))
        self.conn.commit()

    def init_player_data_from_csv(self, csv_file_path):
        """Loads players and their initial data from a CSV file."""
        csv_path = os.path.abspath(csv_file_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")

        with open(csv_path, "r", newline="") as file:
            reader = csv.reader(file)
            headers = next(reader)

            for row in reader:
                player_name = row[headers.index('PlayerName')]
                hour = int(row[headers.index('Hour')])
                consumption = int(row[headers.index('Consumption')])
                production = int(row[headers.index('Production')])

                # Get or insert player ID
                self.cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
                player_id = self.cursor.fetchone()

                if player_id:
                    player_id = player_id[0]
                else:
                    self.insert_player(player_name)
                    self.cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
                    player_id = self.cursor.fetchone()[0]

                # Insert or update the game hour data
                self.cursor.execute('''
                    INSERT OR REPLACE INTO game_hour_data (player_id, game_hour, consumption, production)
                    VALUES (?, ?, ?, ?)
                ''', (player_id, hour, consumption, production))

        self.conn.commit()
        print("CSV player file processing complete")

    def init_battery_data_from_csv(self, csv_filepath):
        """Inserts battery constitution data from a CSV into the database."""
        with open(csv_filepath, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                battery_constitution_json = json.dumps(row)
                player_id = 99  # Placeholder ID (can be randomized if needed)
                game_hour = 0
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                self.cursor.execute('''
                    INSERT INTO battery_log (battery_constitution, player_id, game_hour, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (battery_constitution_json, player_id, game_hour, timestamp))

        self.conn.commit()
        print("Battery log entries inserted successfully!")
        
        
    ############ END OF INITIALISATION METHODS ################## 

    def update_player_game_hour_constitution(self, player_name, game_hour, difference):
        """Updates the player's game hour constitution based on the computed difference."""
        self.cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
        player_id = self.cursor.fetchone()

        if not player_id:
            print(f"Error: Player '{player_name}' not found.")
            return

        player_id = player_id[0]

        # Fetch current constitution, production, and consumption values
        self.cursor.execute('''
            SELECT game_hour_constitution, production, consumption
            FROM game_hour_data
            WHERE player_id = ? AND game_hour = ?
        ''', (player_id, game_hour))

        result = self.cursor.fetchone()
        if result:
            current_constitution = json.loads(result[0]) if result[0] else []
            consumption = result[1]
            production = result[2]
        else:
            current_constitution = []
            production = 0
            consumption = 0

        # Compute the new constitution
        new_constitution = self.calculate_new_constitution(current_constitution, difference, production, consumption)
        new_constitution_json = json.dumps(new_constitution)

        # Update database
        self.cursor.execute('''
            INSERT INTO game_hour_data (player_id, game_hour, game_hour_constitution)
            VALUES (?, ?, ?)
            ON CONFLICT(player_id, game_hour)
            DO UPDATE SET game_hour_constitution = excluded.game_hour_constitution
        ''', (player_id, game_hour, new_constitution_json))

        self.conn.commit()

    def calculate_new_constitution(self, current_constitution, difference, production, consumption):
        """Calculates the new battery constitution based on changes and grid impact."""
        total_difference = sum(-1 if diff.startswith('+') else 1 for diff in difference)

        updated_constitution = current_constitution.copy()

        for diff in difference:
            if diff.startswith('+'):
                letter_to_remove = diff[1]
                if letter_to_remove in updated_constitution:
                    updated_constitution.remove(letter_to_remove)
            elif diff.startswith('-'):
                letter_to_add = diff[1]
                updated_constitution.append(letter_to_add)

        grid_value = total_difference - consumption
        if grid_value < 0:
            updated_constitution.extend(['G'] * abs(grid_value))

        return updated_constitution
    
    def log_battery_data_and_update_constitution(self, player_name, game_hour, new_battery_constitution):
        """Logs battery data, compares logs, and updates game hour constitution."""
        difference = self.compare_battery_log(new_battery_constitution)

        #Get player_id based on player name
        new_battery_constitution_json = json.dumps(new_battery_constitution)
        self.cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
        player_id = cursor.fetchone()[0]
        
        # Insert battery log data
        self.cursor.execute('''
        INSERT INTO battery_log (battery_constitution, player_id, game_hour, timestamp)
        VALUES (?, ?, ?, ?)
        ''', (new_battery_constitution_json, player_id, game_hour, datetime.now()))
        self.conn.commit()
        
        if difference is not None:
            self.update_player_game_hour_constitution(player_name, game_hour, difference)
    
    def compare_battery_log(self,new_battery_constitution):
    
        """Compares the last two battery logs and returns the difference."""
        self.cursor.execute('SELECT battery_constitution FROM battery_log ORDER BY timestamp DESC LIMIT 1')
        logs = self.cursor.fetchall()
        if len(logs) < 1:
            print("Not enough battery logs to compare.")
            return []
        
        # Retrieve the last logged battery constitution (from any player)
        self.cursor.execute('''
        SELECT battery_constitution FROM battery_log
        ORDER BY timestamp DESC
        LIMIT 1
        ''')
        last_log = self.cursor.fetchone()

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


    def print_battery_log(self):
        """Prints all battery log entries."""
        self.cursor.execute('SELECT * FROM battery_log')
        rows = self.cursor.fetchall()
        for row in rows:
            log_id, battery_constitution, player_id, game_hour, timestamp = row
            battery_constitution_list = json.loads(battery_constitution)
            print(f"Log ID: {log_id}, Player ID: {player_id}, Game Hour: {game_hour}, Timestamp: {timestamp}")
            print(f"Battery Constitution: {battery_constitution_list}\n------------")

    def print_player_game_hours(self, player_name):
        """Prints game hour details for a specific player."""
        self.cursor.execute('''
            SELECT g.game_hour, g.consumption, g.production, g.game_hour_constitution
            FROM game_hour_data g
            JOIN players p ON g.player_id = p.player_id
            WHERE p.name = ?
            ORDER BY g.game_hour
        ''', (player_name,))

        rows = self.cursor.fetchall()
        for row in rows:
            game_hour, consumption, production, game_hour_constitution = row
            game_hour_constitution_list = json.loads(game_hour_constitution) if game_hour_constitution else []
            print(f"Game Hour: {game_hour}, Consumption: {consumption}, Production: {production}")
            print(f"Game Hour Constitution: {game_hour_constitution_list}\n------------")


# Example Usage
db = GameDatabase()


