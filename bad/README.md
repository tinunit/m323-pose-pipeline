# bad/: absichtlich falsche Varianten (Runde 2)

Diese Skripte sind **vorgegeben**. Ihr schreibt sie nicht selbst. Sie funktionieren
erst, wenn eure Funktionen aus Runde 1 fertig sind (sie rufen eure Implementierung auf).

Jedes Skript macht genau eine Sache absichtlich falsch. Oben in der Datei stehen
Beobachtungsfragen als Kommentar. Lest sie **vor** dem Start, startet das Skript, und
notiert eure Antworten und eure Erklärung in [bericht.md](../bericht.md) (Teil A, Fragen 2
und 3 beim passenden Schritt).

Start (aus dem Repo-Root, ohne `.py`):

```
uv run python -m bad.b1_aliasing
```

| Abschnitt | Datei | Prinzip |
|---|---|---|
| 3.1 | `b1_aliasing.py` | Immutability (Unveränderlichkeit): zwei Namen, ein Objekt (Aliasing) |
| 3.2 | `b2_pixels_inplace.py` | Pure functions (reine Funktionen): `to_pixels` überschreibt die Koordinaten |
| 3.3 | `b4_stale_cache.py` | Functional core, imperative shell und Memoization: `lru_cache` mit globalem `CONF` |
| 3.4 | `b6_reps.py` | State as a value und Fold: globaler Zähler statt `step`/`count_reps` |

Hinweise:
- `b4` lädt das YOLO-Modell (`yolo11n-pose.pt`) und braucht etwas Zeit.
- Ändert die Skripte nicht. Die Fragen dazu stehen in der Anleitung, Runde 2 (3.1 bis 3.4).
