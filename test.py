from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import random
import time

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
CORS(app)  # Dodanie CORS do obsługi żądań z różnych domen
socketio = SocketIO(app, cors_allowed_origins="*")  # SocketIO z obsługą CORS



@app.route('/game_data', methods=['POST'])
def receive_game_data():
    global game_data
    game_data = request.json
    print(game_data)
    return jsonify({'status': 'success'}), 200


# Funkcja, która losowo wybiera akcje i wysyła je do klienta co kilka sekund
def send_random_action():
    while True:
        action = random.choice([' ', 'no-op'])  # Losowy wybór między spacją a 'no-op'
        socketio.emit('action', {'action': action})  # Wysyłanie akcji do wszystkich klientów
        time.sleep(2)  # Wysyłanie akcji co 2 sekundy

# Endpoint do uruchomienia serwera
@app.route('/')
def index():
    return "Flask server is running."

# Uruchomienie serwera
if __name__ == "__main__":
    socketio.start_background_task(send_random_action)  # Uruchomienie funkcji w tle
    socketio.run(app, port=8080)
