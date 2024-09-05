(function() {
    // Sprawdź, czy gra jest już uruchomiona i czy obiekt gry jest dostępny
    var gameInstance = Runner.instance_;

    if (gameInstance) {
        console.log("Znaleziono instancję gry Dino.");

        // Funkcja do uruchomienia gry
        function startGame() {
            if (gameInstance.playing || gameInstance.crashed) {
                gameInstance.restart();  // Restart, jeśli gra jest zatrzymana lub po kolizji
                console.log("Gra została zrestartowana!");
            } else {
                gameInstance.playIntro();  // Rozpocznij grę
                console.log("Gra została uruchomiona!");
            }
        }
        
        // Funkcja do uruchomienia akcji 'skoku' (jump) dinozaura
        function dinoJump() {
            if (gameInstance.tRex && !gameInstance.tRex.jumping) {
                gameInstance.tRex.startJump(gameInstance.currentSpeed);
                console.log("Dinozaur skacze!");
            }
        }

        // Funkcja do restartowania gry
        function restartGame() {
            if (gameInstance.crashed) {
                gameInstance.restart();
                console.log("Gra została zrestartowana!");
            }
        }

        function sendGameData(obstacle) {
            fetch('http://localhost:8080/game_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    distanceRan: Runner.instance_.distanceRan,
                    type: obstacle.typeConfig.type,
                    speedOffset: obstacle.speedOffset,
                    width: obstacle.width,
                    xPos: obstacle.xPos,
                    crashed: Runner.instance_.crashed
                })
            });
        }

        // Zachowanie oryginalnej funkcji update, aby można było ją wywołać później
        let originalUpdate = gameInstance.update;

        // Dodaj Socket.IO do strony
        var script = document.createElement('script');
        script.src = 'https://cdn.socket.io/4.0.0/socket.io.min.js';
        script.onload = function() {
            var socket = io('http://localhost:8080');  // Komunikacja z lokalnym serwerem

            startGame();
            // Nasłuchiwanie na akcje od serwera
            socket.on('action', function(data) {
                if (data.action === ' ') {
                    console.log('Agent zdecydował: Skok!');
                    dinoJump();  // Wywołaj bezpośrednio funkcję skoku dinozaura
                } else if (data.action === 'no-op') {
                    console.log('Agent zdecydował: no-op!');
                }
            });
        };

        // Nadpisujemy funkcję update, aby wysyłać dane o przeszkodach
        gameInstance.update = function () {
            originalUpdate.apply(this, arguments); // Wywołaj oryginalną funkcję update
        
            // Sprawdzamy, czy są przeszkody
            if (this.horizon && this.horizon.obstacles.length > 0) {
                let obstacle = this.horizon.obstacles[0]; // Pobranie pierwszej przeszkody
                console.log('Obstacle detected:', obstacle); // Wyświetl dane o przeszkodzie w konsoli
                sendGameData(obstacle); // Wysłanie danych do serwera
            }
            restartGame();
        };

        document.head.appendChild(script);
    } else {
        console.log("Nie znaleziono instancji gry Dino.");
    }
})();
