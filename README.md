# Modul 323: Pose-Pipeline (Starter-Repo)

Ihr baut eine Pipeline, die auf Bildern Personen erkennt, Gelenkpunkte (Keypoints) bestimmt, mit Labels vergleicht und Bewegungen in einem Video auswertet, funktional in Python. Zuerst implementieren (Runde 1), dann FP-Prinzipien im eigenen Code entdecken (Runde 2).

**Einstieg: [ANLEITUNG.md](ANLEITUNG.md).** Dort steht alles, vom Setup bis zur Abgabe. Euer Journal ist `journal.md`.

## Setup

```bash
uv sync
```

## Tests und Skripte

```bash
uv run pytest -q                          # alle Tests (anfangs sind die Tests eurer Stubs rot)
uv run pytest tests/test_data.py -q       # Tests eines Schritts
uv run python run_pipeline.py             # Gesamtlauf mit PCK
uv run python visualize.py                # Label und Vorhersage als Bild (out/)
uv run python run_own.py                  # eigene Fotos aus eigene/ (Ausgabe out/eigene/)
uv run python run_squats.py video.mp4 --plot
uv run python -m bad.b1_clean_inplace     # Runde 2: Fehler provozieren
```

Startet alles im Projektordner.

## Arbeitsweise

1. Auf GitHub **Use this template** wählen und ein **privates** Repo erstellen.
2. Die Lehrperson als **Collaborator** einladen.
3. Zu zweit arbeiten, **ein Commit pro Schritt**.
4. Journal laufend ausfüllen.

Eigene Fotos (`eigene/`) und Videos werden **nicht** committed.
