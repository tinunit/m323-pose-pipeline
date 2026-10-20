# bad/: absichtlich falsche Varianten (Runde 2)

Diese Skripte sind **vorgegeben**. Ihr schreibt sie nicht selbst. Sie funktionieren
erst, wenn eure Funktionen aus Runde 1 fertig sind (sie rufen eure Implementierung auf).

Jedes Skript macht genau eine Sache absichtlich falsch. Oben in der Datei stehen
Beobachtungsfragen als Kommentar. Lest sie **vor** dem Start, startet das Skript, und
notiert eure Antworten und eure Erklärung in `journal.md`.

Start (aus dem Repo-Root, ohne `.py`):

```
uv run python -m bad.b1_clean_inplace
```

| Schritt | Datei | Prinzip |
|---|---|---|
| 1 | `b1_clean_inplace.py` | Unveränderlichkeit: Daten in der Liste verändern (Aliasing) |
| 2 | `b2_pixels_inplace.py` | Reine Funktionen: `to_pixels` überschreibt die Koordinaten |
| 3 | `b3_late_binding.py` | Funktionen als Werte: Lambdas in einer Schleife (Closure) |
| 4 | `b4_stale_cache.py` | Reiner Kern, unreine Schale: `lru_cache` mit globalem `CONF` |
| 5 | `b5_average.py` | Fold und Monoid: Mittelwerte statt `Counts` addieren |
| 6 | `b6_reps.py` | Zustand als Wert: globaler Zähler statt `step`/`count_reps` |
| 7 (Bonus) | `b7_flip.py` | Zu schwacher Test: Spiegeln ohne links/rechts zu tauschen |

Hinweise:
- `b4`, `b5` und `b7` laden das YOLO-Modell (`yolo11n-pose.pt`) und brauchen etwas Zeit.
- Ändert die Skripte nicht, um den Fehler zu "reparieren". Die Reparatur steht jeweils
  in der Anleitung (Runde 2).
