# bad/: absichtlich falsche Varianten

Hier schreibt ihr pro Schritt eine Datei, die den Fehler absichtlich provoziert
(Details und Beobachtungsfragen: Anleitung, Abschnitt "Fehler provozieren").
Die Dateien werden nicht vorgegeben. Startet sie mit `uv run python -m bad.<name>`
(ohne `.py`) und notiert eure Beobachtungen in `journal.md`.

| Schritt | Datei | Thema |
|---|---|---|
| 1 | `b1_clean_inplace.py` | Daten direkt in der Liste verändern (Aliasing) |
| 2 | `b2_pixels_inplace.py` | `to_pixels_inplace` überschreibt Koordinaten |
| 3 | `b3_late_binding.py` | Lambdas in einer Schleife (Closure) |
| 4 | `b4_stale_cache.py` | `lru_cache` mit unreiner Funktion (veralteter Cache) |
| 5 | `b5_global_counter.py` | globale Zähler in `ds.map` |
| 5 | `b5_average.py` | Mittelwerte statt `Counts` addieren (nicht assoziativ) |
| 6 | `b6_reps.py` | globaler Zähler, eine Schwelle, zu starke Glättung |
| 7 | `b7_flip.py` | Spiegeln ohne links/rechts, falscher Test, Zufall ohne Seed (Bonus) |

Hinweis: Schritt 5 und Schritt 7 haben mehrere Teilaufgaben, `b6_reps.py` enthält (a), (b) und (c).
