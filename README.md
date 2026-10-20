# Modul 323: Pose-Pipeline (Starter-Repo)

Ihr baut eine Pipeline, die auf Bildern Personen erkennt, Gelenkpunkte (Keypoints)
bestimmt, mit Labels vergleicht und Kniebeugen in einem Video zählt, funktional in Python.

Die vollständige Aufgabenstellung steht in der **Anleitung**
(`m323_pose_pipeline_anleitung.pdf`, von der Lehrperson). Dieses Repo enthält
das Gerüst: Funktionen mit Signatur und Beschreibung, die ihr implementiert
(`raise NotImplementedError`), die vorgegebenen Tests und das Journal-Template.

## Setup

Python 3.11 oder neuer und [uv](https://docs.astral.sh/uv/) werden benötigt. Kein `pip`.

```
uv sync
```

Das erzeugt `.venv/` und installiert alle Pakete (ultralytics, datasets, hypothesis, opencv, pytest),
etwa 1.5 GB. Unter Linux wird automatisch die CPU-Version von torch installiert (ohne CUDA).
Der Datensatz COCO8-pose liegt schon in `data/coco8-pose/`. Das Modell `yolo11n-pose.pt`
wird beim ersten Aufruf automatisch geladen (ca. 6 MB, läuft auf der CPU).

## Tests und Skripte

```
uv run pytest                          # alle Tests
uv run pytest tests/test_data.py       # nur ein Schritt
uv run python -m bad.b1_clean_inplace  # eine eurer "bad"-Varianten
uv run python run_pipeline.py          # ganze Pipeline
uv run python run_squats.py video.mp4  # Kniebeugen zählen
```

Am Anfang schlagen die Tests mit `NotImplementedError` fehl. Das ist so gewollt:
Ziel ist, dass sie Schritt für Schritt grün werden.

## Struktur

```
pipeline/   types.py (gegeben), data, geometry, draw, filters, inference, evaluate, reps, augment
bad/        eure absichtlich falschen Varianten (siehe bad/README.md)
tests/      vorgegebene Tests + test_eigene.py (mindestens ein eigener Test)
run_pipeline.py, run_squats.py, journal.md
```

Was bereits vorgegeben ist, ist im Code mit `# vorgegeben` markiert. Alles andere implementiert ihr.

## Arbeitsweise

1. Auf GitHub **„Use this template“ → „Create a new repository“** wählen, Sichtbarkeit **Private**,
   Lehrperson als Collaborator hinzufügen, dann klonen. (Ein Fork wäre öffentlich, dann sieht
   die ganze Klasse euren Code.) Optional ein Branch pro Schritt, z.B. `schritt-2`.
2. Pro Schritt: FP-Input lesen, Funktionen in `pipeline/` implementieren, Fehler in `bad/` provozieren,
   im `journal.md` erklären, Tests grün machen, eigenen Test ergänzen.
3. Pro Schritt committen, z.B. `git commit -m "Schritt 2: Geometrie"`.
4. Eigene Fotos und Videos gehören nicht ins Repo (`eigene/`, `*.mp4` sind in `.gitignore`).

## Abgabe (Anleitung, Abschnitt 11)

- Code in `pipeline/`, `bad/`, `tests/`: `uv run pytest` ist grün,
  `run_pipeline.py` läuft vollständig, `run_squats.py video.mp4` gibt die Anzahl Kniebeugen aus.
- `journal.md`: pro Schritt 3 bis 6 Sätze zu den drei Fragen.
- Präsentation (5 Minuten): ein Schritt nach Wahl, Fehler live zeigen, Lösung erklären.
- Bewertung: Pipeline und Tests 40 %, Journal 40 %, Präsentation 20 %. Schritt 7 gibt Bonuspunkte.
