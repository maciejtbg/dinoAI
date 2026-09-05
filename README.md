# dinoAI — agent RL grający w Dino z Chrome

Agent uczenia ze wzmocnieniem (DQN), który uczy się grać w offline'ową grę "Dino" z przeglądarki Chrome, obserwując ekran gry i sterując klawiaturą.

## Jak to działa

1. `mss` przechwytuje zrzuty wybranego fragmentu ekranu (obszar gry).
2. OpenCV i `pytesseract` (OCR) przetwarzają obraz i wykrywają moment zakończenia gry (napis "game over").
3. Środowisko zbudowane jest na `gymnasium` (przestrzeń obserwacji jako obraz, przestrzeń akcji: skok / kucnięcie / brak akcji).
4. `pydirectinput` wysyła naciśnięcia klawiszy symulujące sterowanie dinozaurem.

## Stack

Python, OpenCV, Gymnasium (Reinforcement Learning), mss, pytesseract, pydirectinput.
