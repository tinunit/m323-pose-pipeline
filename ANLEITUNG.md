# Modul 323: Pose-Pipeline

Funktionale Programmierung (FP) ist nicht an Haskell gebunden: Ihre Prinzipien wie unveränderliche Daten, pure functions und Funktionen als Werte stecken in fast allen modernen Sprachen und Frameworks. In diesem Projekt lernt ihr sie an einem konkreten Beispiel kennen: einer kleinen Computer-Vision-Pipeline in Python, die Personen und ihre Gelenke auf Bildern und in Videos erkennt. In **Runde 1** baut ihr die Pipeline, in **Runde 2** findet ihr die FP-Prinzipien in eurem eigenen Code und in der Software, die ihr in eurem Lehrbetrieb entwickelt.

## Los geht's: YOLO in 5 Zeilen

Ein trainiertes Modell zu benutzen, ist erstaunlich wenig Code. Probiert es aus, bevor ihr weiterlest.

> **Vorher:** Repo eingerichtet und `uv sync` ausgeführt? Sonst zuerst [Repo einrichten](#repo-einrichten-einmal-pro-team) und [Setup prüfen](#setup-prüfen).

Legt im Projektordner eine Datei `erste_schritte.py` an und startet sie mit `uv run python erste_schritte.py`.

**Beispiel 1: Was ist auf dem Bild?** (Klassifikation)

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")                     # Modell laden (wird beim ersten Mal heruntergeladen)
result = model("data/coco8-pose/images/train/000000000036.jpg", verbose=False)[0]
print("Das ist:", result.names[result.probs.top1], f"({result.probs.top1conf:.0%} sicher)")
result.save("out/klassifikation.jpg")              # Bild mit Ergebnis speichern
```

Ausgabe:

```
Das ist: umbrella (100% sicher)
```

**Beispiel 2:** Ändert nur den Bildpfad auf `data/coco8-pose/images/val/000000000110.jpg`. Ausgabe: `Das ist: pizza (50% sicher)`.

**Beispiel 3: Wo sind die Personen und ihre Gelenke?** (Pose)

```python
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")                    # nur der Modellname ist anders
result = model("data/coco8-pose/images/val/000000000113.jpg", verbose=False)[0]
print("Personen gefunden:", len(result.boxes))
print("Nase der ersten Person (x, y):", result.keypoints.xy[0][0].int().tolist())
result.save("out/pose.jpg")
```

Ausgabe:

```
Personen gefunden: 3
Nase der ersten Person (x, y): [239, 129]
```

![Die drei Beispiele: Schirm, Pizza, Pose](docs/img/erste_schritte.jpg)

Öffnet die Bilder in `out/`. Statt `result.save(...)` zeigt `result.show()` das Bild direkt an.

**Probiert selbst:**

- Ein eigenes Foto (Haustier, etwas auf dem Tisch) mit dem Klassifikationsmodell. Am besten klappt es mit **einem** grossen Gegenstand in der Mitte.
- `val/000000000136.jpg` mit dem Klassifikationsmodell. Was sagt es, und was ist wirklich auf dem Bild? Warum liegt es daneben? (Tipp: Das Modell kennt nur 1000 feste Kategorien.)

**Merkt euch für später:** Beispiel 3 liefert genau die Daten, mit denen wir arbeiten: pro Person eine Box und 17 Punkte mit Koordinaten. Kapitel 1 zeigt, wie diese Daten aufgebaut sind.

---

## Inhalt

- [Los geht's: YOLO in 5 Zeilen](#los-gehts-yolo-in-5-zeilen)
0. [Worum es geht](#0-worum-es-geht)
1. [YOLO und Pose](#1-yolo-und-pose)
2. [Runde 1: Implementieren](#2-runde-1-implementieren)
   - [Schritt 1: Daten laden](#schritt-1-daten-laden) · [2: Geometrie](#schritt-2-geometrie) · [3: YOLO](#schritt-3-yolo) · [4: Video](#schritt-4-video) (4a Kniebeugen, 4b eigene Analyse)
3. [Runde 2: FP-Prinzipien](#3-runde-2-fp-prinzipien)
   - [3.0 FP ist überall](#30-fp-ist-überall) · [3.1 Immutability und Aliasing](#31-immutability-und-aliasing) · [3.2 Pure functions](#32-pure-functions) · [3.3 Functional core, imperative shell, Memoization](#33-functional-core-imperative-shell-und-memoization) · [3.4 State as a value und Fold](#34-state-as-a-value-und-fold)
4. [Bericht und Abgabe](#4-bericht-und-abgabe)
5. [Spickzettel](#5-spickzettel)

---

## 0. Worum es geht

**Die Pipeline in einem Satz:** Bild rein, Personen mit 17 Gelenkpunkten raus, daraus Winkel berechnen und Bewegungen zählen.

![Überblick über die Pipeline](docs/img/pipeline_ueberblick.svg)

Im Bild seht ihr sieben Stationen. Ihr programmiert vier davon (Schritte 1, 2, 4 und 6 im Bild, bei uns **Schritt 1 bis 4**). Die Stationen Filtern (3), Auswerten (5) und Spiegeln (7) im Bild sind bereits **fertig vorgegeben**.

### Der Datensatz: COCO

**COCO** ("Common Objects in Context") ist ein grosser, öffentlicher Bilddatensatz mit über 200'000 Bildern von Alltagsszenen. Bei vielen Personen haben Menschen von Hand 17 Körperpunkte markiert (Nase, Schultern, Ellbogen, ...). Wir nutzen **COCO8-Pose**: nur 8 Bilder davon (4 `train`, 4 `val`), klein genug für den Laptop.

So liegen die Daten im Repo:

```
data/coco8-pose/
├── images/
│   ├── train/   000000000036.jpg, ...
│   └── val/     ...
└── labels/
    ├── train/   000000000036.txt, ...
    └── val/     ...
```

Bild und Label-Datei haben **denselben Namen** (`.jpg` bzw. `.txt`) und liegen im gleichen Unterordner (`train` oder `val`). Pro Person gibt es eine Zeile. (Die `.cache`-Dateien in `labels/` könnt ihr ignorieren.)

**Beispiel:** `train/000000000036.jpg` zeigt eine Person mit Schirm, also hat `labels/train/000000000036.txt` genau eine Zeile.

![Bild 000000000036 mit Box, drei markierten Punkten und der passenden Label-Zeile](docs/img/coco_beispiel.png)

```
0 0.671279 0.617945 0.645759 0.726859 0.519751 0.381250 2.000000 0.550936 0.348438 2.000000 ... 0.000000 0.000000 0.000000
```

Die erste Zahl `0` bedeutet "Person", dann folgt die Box, dann 17 × (`x y v`). Genau erklärt in [Kapitel 1: Eine Label-Zeile zerlegt](#eine-label-zeile-zerlegt).

Diese Labels wurden **von Menschen von Hand markiert**: Jemand hat jeden Körperpunkt einzeln angeklickt. Sie gelten als die **richtige Antwort** (englisch *ground truth*), an der wir das Modell messen.

Hier liegen die Labels (gelb) und die Vorhersage von YOLO (cyan) auf demselben Bild übereinander:

![Label (gelb) und YOLO-Vorhersage (cyan) übereinander, mit Zoom auf den Kopf](docs/img/label_vs_yolo_overlay.png)

YOLO kommt den Labels sehr nahe (im Schnitt etwa 12 Pixel Abweichung bei einem 640 Pixel breiten Bild), aber nicht exakt.

### Wie gut ist das Modell? (kurz)

Ein vorhergesagter Punkt zählt als **richtig**, wenn er näher beim Label-Punkt liegt als eine **Toleranz** (10 % der Personengrösse). **PCK** ist der Anteil der richtigen Punkte. Für unsere 4 `val`-Bilder liegt er bei etwa **0.78** (90 von 116 Punkten).

![PCK: richtig, wenn der vorhergesagte Punkt im Toleranzkreis liegt](docs/img/pck_toleranz.svg)

Das ist nur zum Verstehen, ihr müsst nichts davon rechnen oder auswendig können.

### Daten und Modell

Wir arbeiten mit zwei Quellen, die **dieselbe Art von Daten** liefern:

| | Testdaten mit Labels | Inferenz mit YOLO |
|---|---|---|
| Woher | **COCO8-Pose** (siehe oben), im Repo unter `data/coco8-pose/` | das vortrainierte Modell **YOLO11n-Pose** (`yolo11n-pose.pt`) |
| Was | Pro Bild eine Textdatei: Menschen haben für jede Person die 17 Gelenkpunkte von Hand markiert. Das ist die "richtige Antwort" | Das Modell bekommt nur das Bild und sagt die 17 Gelenkpunkte pro Person **vorher** |

Beide liefern pro Person dasselbe: eine Box und 17 Punkte. Darum können wir Labels und Vorhersagen direkt vergleichen. Kapitel 1 zeigt, wie ähnlich die beiden Formate sind.

### Die zwei Runden

In Runde 1 schreibt ihr nur die **Kernstücke**, der Rest liegt fertig im Repo. In Runde 2 macht ihr vier Prinzipien mit fertigen `bad/`-Skripten **kaputt** und erklärt, was ihr seht.

### Zeitplan (Richtwert, zusammen ca. 4 Stunden)

| Teil | Zeit |
|---|---|
| Setup + "Los geht's" + Kapitel 1 (YOLO ausprobieren, Daten verstehen) | 30 min |
| Runde 1: Schritte 1 bis 4a | 1 h 15 |
| Runde 2: Einstieg + Recherche im Betrieb | 30 min |
| Runde 2: FP-Prinzipien (4 Abschnitte) | 45 min |
| Bericht fertig machen | 15 min |
| Puffer | 25 min |
| **Zum Schluss:** Schritt 4b, eure eigene Analyse | so weit ihr kommt |

### Teams

Ihr arbeitet in Teams von **2 bis 4 Personen**. Teilt die 4 Schritte unter euch auf, aber alle müssen jeden Schritt erklären können. Die Recherche im Betrieb (Kapitel 3.0) macht **jede Person** aus Sicht des eigenen Lehrbetriebs.

### Voraussetzungen

- Pro Person ein **GitHub-Konto**.
- **git** und **uv** installiert. uv installieren:
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Repo einrichten (einmal pro Team)

1. **Eine Person** des Teams öffnet dieses Repo auf GitHub und klickt **"Use this template" → "Create a new repository"**.
2. Name z.B. `m323-pose-<eure-namen>`, Sichtbarkeit **Private**, dann **Create repository**.
3. Im neuen Repo: **Settings → Collaborators → Add people**:
   - **alle anderen Teammitglieder**,
   - die **Lehrperson** (GitHub: `tinunit`).
4. Alle klonen das neue Repo und richten es ein:

   ```bash
   git clone git@github.com:<euer-konto>/<euer-repo>.git
   cd <euer-repo>
   uv sync
   ```

**Nicht forken:** Ein Fork eines öffentlichen Repos ist selbst öffentlich, dann sieht die ganze Klasse euren Code. Mit "Use this template" bekommt ihr ein eigenes, privates Repo.

**Zusammenarbeiten:** Vor dem Arbeiten `git pull`, danach `git add`, `git commit`, `git push`. Arbeitet in verschiedenen Dateien, so gibt es keine Konflikte. Ein Commit pro Schritt, z.B. `git commit -m "Schritt 2: Geometrie"`.

### Setup prüfen

```bash
uv sync                # installiert alles (einmalig, dauert ein paar Minuten)
uv run pytest -q       # alle Tests: gegebener Code grün, eure Stubs rot
```

Am Anfang sind nur die Tests **eurer** Funktionen rot. Euer Ziel pro Schritt: die zugehörige Testdatei grün. Der Datensatz **COCO8-Pose** (8 Bilder) ist bereits im Repo. Das Modell `yolo11n-pose.pt` (6 MB) wird beim ersten Aufruf automatisch heruntergeladen. Alle Befehle startet ihr im Projektordner.

### Arbeiten mit KI

- KI-Assistenten sind **erlaubt und erwünscht**: Funktionen erklären lassen, Fehlermeldungen deuten, Tests erläutern.
- **Verstehen, was ihr übernehmt.** Die Erklärungen im Bericht müssen von euch kommen.
- Gebt der KI die Docstrings und den Test mit: Sie liefert dann viel bessere Vorschläge.
- Im Bericht (Teil C) beschreibt ihr: **Wo hat die KI geholfen? Wo lag sie falsch?**

---

## 1. YOLO und Pose

Ihr braucht kein Wissen über neuronale Netze, nur ein Bild davon, was aus dem Modell herauskommt. Ausprobiert habt ihr YOLO schon ([Los geht's](#los-gehts-yolo-in-5-zeilen)), jetzt schauen wir genauer hin.

### Was ist YOLO?

YOLO ("You Only Look Once") ist ein Modell, das ein Bild **einmal** ansieht und direkt Ergebnisse liefert: Boxen, Punkte, Masken. Wir benutzen ein fertig trainiertes Modell. Trainiert wird hier nichts.

### Vier Arten von Aufgaben

![Klassifikation, Detektion, Segmentierung, Pose](docs/img/yolo_aufgaben.png)

| Aufgabe | Frage ans Bild | Ausgabe |
|---|---|---|
| Klassifikation | Was ist das? | **eine** von 1000 festen Kategorien |
| Detektion | Was ist wo? | Boxen mit Klasse und Konfidenz |
| Segmentierung | Welche Pixel gehören dazu? | Masken |
| **Pose** (unsere) | Wo sind die Gelenke? | Box + 17 Punkte pro Person |

Die Klassifikation **muss** sich für eine feste Kategorie entscheiden. Zeigt man ihr ein Bild, das in keine passt, rät sie trotzdem, z.B. "crossword_puzzle". Merkt euch das: Modelle sagen immer etwas, auch wenn sie falsch liegen.

### Die 17 Keypoints

![Nummerierte Keypoints](docs/img/keypoints_nummeriert.png)

| Idx | Name | Idx | Name | Idx | Name |
|---|---|---|---|---|---|
| 0 | Nase | 6 | rechte Schulter | 12 | rechte Hüfte |
| 1 | linkes Auge | 7 | linker Ellbogen | 13 | linkes Knie |
| 2 | rechtes Auge | 8 | rechter Ellbogen | 14 | rechtes Knie |
| 3 | linkes Ohr | 9 | linkes Handgelenk | 15 | linkes Fussgelenk |
| 4 | rechtes Ohr | 10 | rechtes Handgelenk | 16 | rechtes Fussgelenk |
| 5 | linke Schulter | 11 | linke Hüfte | | |

Es gibt **keine Mund-Punkte**: Das Gesicht besteht aus Nase, Augen und Ohren. "Links" und "rechts" sind aus Sicht der Person, nicht des Betrachters. Für den Kniewinkel brauchen wir Hüfte, Knie, Fussgelenk: **11, 13, 15** (links) bzw. **12, 14, 16** (rechts).

### Eine Label-Zeile zerlegt

Pro Person eine Zeile in `data/coco8-pose/labels/*/<bild>.txt`. Hier die erste Person aus `000000000113.txt` (der Mann in den Abbildungen), Zahlen auf 3 Stellen gekürzt:


![Eine Label-Zeile, Feld für Feld zerlegt](docs/img/label_zeile.svg)

Aufbau: `Klasse | cx cy w h | (x y v) × 17` = 1 + 4 + 51 = **56 Zahlen**.

| Feld | Bedeutung |
|---|---|
| Klasse | immer `0` (Person) |
| `cx cy w h` | Box-Mitte und Grösse |
| `x y` | Position des Punkts |
| `v` | Sichtbarkeit: **0** = nicht markiert, **1** = markiert aber verdeckt, **2** = markiert und sichtbar |

Alle Positionen und Grössen sind **normalisiert** (0 bis 1): `x` ist der Anteil der Bildbreite, `y` der Anteil der Bildhöhe. Für Pixel multipliziert man mit Breite bzw. Höhe (Schritt 2). Im Beispiel ist Punkt 3 (linkes Ohr) `0 0 0`: nicht markiert, die Position ist bedeutungslos.

### Koordinaten: normalisiert und Pixel

![Pixel-Koordinaten und normalisierte Koordinaten](docs/img/koordinaten.png)

**Das Koordinatensystem von Bildern:** Der Ursprung `(0, 0)` liegt **oben links**. `x` wächst nach rechts, **`y` wächst nach unten** (anders als in der Mathematik).

| | Pixel | Normalisiert |
|---|---|---|
| Einheit | Bildpunkte | Anteil der Breite bzw. Höhe |
| Bereich | `x`: 0 bis Breite, `y`: 0 bis Höhe | `x` und `y`: 0 bis 1 |
| Beispiel (Ellbogen R oben, Bild 416 × 640) | `(19, 260)` | `(19 / 416, 260 / 640)` = `(0.046, 0.406)` |
| Wo | zum Zeichnen und Rechnen von Winkeln | in den Label-Dateien und in der YOLO-Ausgabe (`xywhn`, `xyn`) |

Umrechnen (das macht `to_pixels` in Schritt 2):

```python
x_px = x * breite        # auch cx und w der Box
y_px = y * hoehe         # auch cy und h der Box
```

**Warum normalisiert?** Die Zahlen hängen nicht von der Bildgrösse ab. Ein Label passt zum Bild, egal ob es in 416 × 640 oder verkleinert vorliegt.

**Warum Winkel in Pixeln?** Normalisiert wird jedes Bild auf ein Quadrat 1 × 1 "gequetscht" (rechte Hälfte der Abbildung): Ein hohes Bild wird in der Höhe stärker gestaucht als in der Breite. Strecken kippen, und Winkel stimmen nicht mehr. Im Beispiel: Ellbogenwinkel **146.8°** in Pixeln, aber **130.7°** normalisiert gerechnet.

**Die Box:** `cx cy` ist die **Mitte** der Box, nicht die Ecke oben links. Die Ecke oben links ist `(cx - w/2, cy - h/2)`.

### Die YOLO-Ausgabe hat denselben Aufbau

```python
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")                          # Modell laden (lädt Gewichte)
results = model.predict(images, conf=0.25, verbose=False, device="cpu")  # images: Liste von PIL-Bildern
for r in results:                                        # ein Ergebnis pro Bild
    r.boxes.xywhn        # (n, 4)    Boxen cx cy w h, normalisiert   ← wie im Label
    r.boxes.conf         # (n,)      Konfidenz pro Person
    r.keypoints.xyn      # (n, 17, 2) Punkte x y, normalisiert        ← wie im Label
    r.keypoints.conf     # (n, 17)   Konfidenz pro Punkt              ← entspricht v
```

`n` ist die Anzahl erkannter Personen (kann 0 sein). Die Tensoren wandelt ihr mit `.tolist()` in normale Listen um.

![Label und YOLO-Vorhersage im Vergleich](docs/img/label_vs_yolo.png)

| Label-Zeile | YOLO-Ausgabe |
|---|---|
| `cx cy w h` | `r.boxes.xywhn[i]` |
| `x y` pro Punkt | `r.keypoints.xyn[i][j]` |
| `v` pro Punkt | `r.keypoints.conf[i][j]` |
| (immer 1.0) | `r.boxes.conf[i]` (Konfidenz der Person) |

**Zwei Unterschiede**, die später wichtig werden:

1. **`v` ist nicht gleich `conf`.** Im Label ist `v` eine Sichtbarkeit (0/1/2), in der Vorhersage eine Konfidenz (0.0 bis 1.0, praktisch nie genau 0). "Markiert, wenn `v > 0`" gilt für Labels, **nicht** für Vorhersagen.
2. **YOLO liefert immer alle 17 Punkte.** Auch verdeckte oder ausserhalb des Bildes liegende Punkte werden geraten, nur mit tiefer Konfidenz.

### Unsere Datentypen

In `pipeline/types.py`, unveränderlich (`NamedTuple`):

```python
class Keypoint(NamedTuple):      # z.B. die Nase: Keypoint(x=0.520, y=0.381, v=2.0)
    x: float                       # Anteil der Bildbreite: 0 = links, 1 = rechts (0.520 = etwas rechts der Mitte)
    y: float                       # Anteil der Bildhöhe: 0 = oben, 1 = unten (0.381 = im oberen Drittel)
    v: float                       # Label: 0 = nicht markiert, 1 = verdeckt, 2 = sichtbar. Vorhersage: Konfidenz 0..1

class Person(NamedTuple):
    box: tuple[float, float, float, float]   # cx, cy, w, h
    keypoints: tuple[Keypoint, ...]          # immer genau 17
    score: float = 1.0                       # Label: 1.0. Vorhersage: Konfidenz
```

---

## 2. Runde 1: Implementieren

Runde 1 hat **vier Schritte**. Pro Schritt: Ziel, Gegeben, **eure Aufgabe**, Hinweis, Prüfen. Die Docstrings beschreiben genau, was jede Funktion tun soll. Ihr schreibt nur die `# TODO`-Stellen. Alles andere in `pipeline/` ist fertig.

Allgemeine Regeln: **Funktionen verändern ihre Argumente nicht**, sie geben neue Werte zurück. Verwendet die Typen `Keypoint` und `Person`. In den Docstrings steht **Pure.** (pure function, reine Funktion: nur Eingabe, nur Ausgabe) oder **Impure.** (hat Nebenwirkungen, *side effects*: Dateien, Modell, Video).

### Schritt 1: Daten laden

**Ziel:** Label-Dateien einlesen und in `Person`-Objekte umwandeln. **FP-Thema:** Immutability (Unveränderlichkeit): `Person` und `Keypoint` sind `NamedTuple`s und lassen sich nach dem Bauen nicht mehr ändern.

**Gegeben** (`pipeline/data.py`): `parse_label_file` (ruft eure Funktion für jede Zeile auf), `FEATURES`, `person_to_row`, `row_to_person`, `labelled_count`, `load_split`, `load_all`, `load_or_build`, `summary`. Sie bauen aus euren Personen ein Hugging-Face-`Dataset` und speichern es auf der Festplatte.

**Eure Aufgabe** (`pipeline/data.py`):

| Funktion | Eingabe | Ausgabe | Was passiert |
|---|---|---|---|
| `parse_label_line(line)` | **eine Zeile** als `str`, z.B. `"0 0.193 0.405 … 0.151 0.911 1"` | **eine** `Person`, oder `None` | 56 Zahlen → Box + 17 `Keypoint` |

Der Anfang ist schon da: Zeile aufteilen, Anzahl prüfen, in Zahlen umwandeln (`nums`, ein Tupel mit 56 Zahlen), und am Schluss `return Person(box=box, keypoints=kps)`. **Ihr ergänzt nur `box` und `kps`** und löscht die Zeile `raise NotImplementedError(...)`.

Das Lesen der Datei von der Festplatte (`Path.read_text()`) und das Bauen des Datasets erledigt das gegebene `load_split`. Eure Funktion bekommt nur **Text** und gibt **Werte** zurück.

**Hinweis:**

```python
# nums = (0.0, 0.67, 0.62, 0.65, 0.73, 0.52, 0.38, 2.0, 0.55, 0.35, 2.0, ...)
#          │    └──── box ─────────┘  └─ Keypoint 0 ─┘ └─ Keypoint 1 ─┘
#        Klasse
# Slicing: nums[a:b] gibt die Elemente a bis b-1 als Tupel
# Keypoint i steht an nums[5 + 3*i : 8 + 3*i]   ->  Keypoint(x, y, v)
# 17 Keypoints als Tupel: tuple(... for i in range(17))
```

**Prüfen:**

```bash
uv run pytest tests/test_data.py -q
uv run python -m pipeline.data      # Zusammenfassung pro Bild
```

Ihr solltet für COCO8 pro Split 4 Bilder sehen. Erwartung: alle Tests in `test_data.py` grün.

### Schritt 2: Geometrie

**Ziel:** Normalisierte Koordinaten in Pixel umrechnen und den Kniewinkel bestimmen. **FP-Thema:** pure functions (reine Funktionen): `to_pixels` gibt eine **neue** `Person` zurück und verändert die Eingabe nicht.

**Gegeben** (`pipeline/geometry.py`): `KNEE_JOINTS` (Indizes von Hüfte, Knie, Fussgelenk pro Seite) und **`angle_at(a, b, c)`**: der Winkel in Grad bei `b` zwischen den Strecken `b→a` und `b→c` (0 bis 180), oder `None`, wenn eine Strecke Länge 0 hat. Dazu `pipeline/draw.py` komplett (`draw_skeleton` zeichnet ein Skelett auf ein Bild).

**Eure Aufgabe** (`pipeline/geometry.py`):

| Funktion | Eingabe | Ausgabe | Was passiert |
|---|---|---|---|
| `to_pixels(p, width, height)` | **eine** `Person` (normalisiert, 0..1) + Bildbreite und -höhe in Pixeln (`int`) | **eine neue** `Person` in Pixeln | `x * width`, `y * height` für alle 17 Punkte (Box ist schon umgerechnet). `v` und `score` bleiben |
| `knee_angle(p_px, side)` | **eine** `Person` in Pixeln + `"left"` oder `"right"` | Kniewinkel in Grad (`float`), oder `None` | `None`, wenn einer der drei Punkte (Hüfte, Knie, Fussgelenk) nicht markiert ist (`v <= 0`), sonst `angle_at` aufrufen |

Gestreckt ≈ 180°, tiefe Kniebeuge ≈ 70–90°.

In `to_pixels` ist die Box schon umgerechnet (`box = ...`). Ihr ergänzt `kps` und löscht die Zeile `raise NotImplementedError(...)`. In `knee_angle` stehen die drei Punkte schon bereit (`hip, knee, ankle`).

**Hinweis:**

```python
# Ein Keypoint ist ein NamedTuple: k.x, k.y, k.v. Einen neuen Keypoint baut man mit Keypoint(x, y, v).
# Ein Tupel aus einer Schleife: tuple(f(k) for k in p.keypoints)
# knee_angle: früh aussteigen mit  if <Bedingung>: return None
# hip.v, knee.v, ankle.v sind die Sichtbarkeiten der drei Punkte
```

Gerechnet wird **in Pixeln**, nicht in normalisierten Koordinaten (siehe [Koordinaten](#koordinaten-normalisiert-und-pixel) in Kapitel 1: 146.8° statt 130.7°). Darum gibt es `to_pixels` vor `knee_angle`.

**Prüfen:**

```bash
uv run pytest tests/test_geometry.py -q
uv run python -m pipeline.geometry     # speichert ein Beispielbild mit Skelett in out/
```

Öffnet das Bild in `out/` und prüft, ob das Skelett auf der Person sitzt.

### Schritt 3: YOLO

**Ziel:** Das Modell laden und seine Ausgabe in unsere `Person`-Typen übersetzen. Teil C: eigenes Foto. **FP-Thema:** functional core, imperative shell (reiner Kern, unreine Schale) und memoization (Zwischenspeichern von Ergebnissen).

**Gegeben** (`pipeline/inference.py`): `FEATURES_WITH_PRED`, `WEIGHTS`, `KP_CONF`, `_persons_of` (liest die vier Felder aus dem YOLO-Ergebnis und ruft `result_to_persons` auf), `predict_batch`, `predict_split`, `predict_image`, `confident` (Punkte mit Konfidenz `< 0.5` bekommen `v = 0`, damit `knee_angle` nicht für **geratene** Knie einen Winkel liefert) und `annotate` mit der Kommandozeile. Dazu `run_own.py` und `visualize.py` komplett.

So hängen die Funktionen zusammen:

![Datenfluss in predict_batch: vom Bild zur neuen Spalte pred](docs/img/yolo_datenfluss.svg)

**Eure Aufgabe** (`pipeline/inference.py`):

| Funktion | Eingabe | Ausgabe | Was passiert |
|---|---|---|---|
| `get_model(weights)` | Dateiname der Gewichte (`str`), z.B. `"yolo11n-pose.pt"` | das **Modell** (`YOLO`) | das Modell laden. Der Decorator `@lru_cache` darüber sorgt dafür, dass es nur **einmal** geladen wird |
| `result_to_persons(boxes_xywhn, boxes_conf, kps_xyn, kps_conf)` | **vier Listen** für die `n` erkannten Personen: `n × 4` Box-Zahlen, `n` Konfidenzen, `n × 17 × 2` Punkt-Koordinaten, `n × 17` Punkt-Konfidenzen (oder `None`) | **Tupel von Personen**, Länge `n`; keine Erkennung → `()` | pro Person Box, 17 `Keypoint(x, y, konfidenz)` und `score`. Die Schleife und der Fall `kps_conf is None` (dann `v = 1.0`) sind schon da, ihr ergänzt `kps` |

`result_to_persons` ist **pure**: Es bekommt nur Zahlenlisten und kennt keine YOLO-Objekte. Darum lässt es sich ohne Modell testen. Das ist der *Kern*. `get_model`, `predict_batch` und `predict_image` sind die *Schale*: Sie laden das Modell und berühren Bilder.

**Hinweis:** Die API aus Kapitel 1:

```python
model = YOLO("yolo11n-pose.pt")        # wie in "Los geht's": eine Zeile

# zip läuft über mehrere Listen gleichzeitig; Paare lassen sich direkt auspacken:
for (x, y), v in zip([[0.5, 0.2], [0.6, 0.3]], [0.9, 0.8]):
    print(x, y, v)                     # 0.5 0.2 0.9  /  0.6 0.3 0.8
```

Probiert es zuerst interaktiv aus (`uv run python`), mit einem Bild aus `data/coco8-pose/images/val/`, und schaut euch die vier Felder an. So seht ihr, wie wenig Code das Modell braucht.

**Prüfen (Teil A und B):**

```bash
uv run pytest tests/test_inference.py -q
uv run python -m pipeline.inference data/coco8-pose/images/val/000000000110.jpg
```

Der zweite Befehl schreibt `out/annotated.png` und druckt die Kniewinkel. Beim ersten Mal wird das Modell geladen, das dauert einen Moment. Das Bild muss ein Skelett auf der Person zeigen. Mit `uv run python visualize.py` bekommt ihr zusätzlich Label und Vorhersage nebeneinander in `out/vis_val.png`.

**Teil C: eigenes Foto**

1. Legt ein Foto in den Ordner `eigene/` (Ordner anlegen, falls er fehlt). Am besten **von der Seite**, **ganzer Körper** im Bild, gut beleuchtet. Handyfotos sind in Ordnung: Drehung (EXIF), Farbformat und Grösse werden von `run_own.py` automatisch korrigiert.
2. ```bash
   uv run python run_own.py
   ```
3. Ergebnisse (Skelett und Kniewinkel) liegen in `out/eigene/`.

Fragen für euch: Funktioniert die Erkennung gut? Wie ändert sich der Winkel, wenn das Knie verdeckt ist? Eigene Fotos kommen **nicht** ins Repo.

### Schritt 4: Video

**Ziel:** Bewegungen in Videos auswerten. Zuerst an einem festen Beispiel, den **Kniebeugen** (4a). Ganz am Schluss, nach Runde 2, untersucht ihr eine Bewegung, die **euch** interessiert (4b). **FP-Thema:** state as a value (Zustand als Wert, ein *Reducer*) und fold (Falten einer Liste zu einem Wert).

#### 4a: Beispiel Kniebeugen zählen

Filmt euch gegenseitig: etwa 10 Kniebeugen, 10 bis 20 Sekunden. Legt die Videos in einen Ordner `videos/` im Projekt (Ordner anlegen, er steht in der `.gitignore`, die Videos kommen also nicht ins Repo). Tipps:

- **Kopf UND Füsse** im Bild, 3 bis 4 m Abstand.
- **Genau von der Seite** filmen.
- Kamera ruhig (Stativ oder ablegen).
- Nicht am Anfang oder Ende ins Bild hinein- oder hinauslaufen (oder das wegschneiden).

**Gegeben** (`pipeline/reps.py`): `RepState`, `START`, die Schwellen (`DOWN_BELOW`, `UP_ABOVE`), `frames` (Generator, ein Bild nach dem anderen), `angles` (Kniewinkel pro Bild), `drop_missing`, `ema` (glättet die Winkel), `states`. Und die erste Regel von `step`. `run_squats.py` komplett.

**Eure Aufgabe** (`pipeline/reps.py`):

| Funktion | Eingabe | Ausgabe | Was passiert |
|---|---|---|---|
| `step(state, angle)` | **ein** `RepState` (z.B. `RepState("up", 3)`) + **ein** Winkel (`float`) | **ein neuer** `RepState` | ein Schritt des Zustandsautomaten. Ihr ergänzt die Rückkehr von `down` nach `up` (mit `count + 1`) und den Fall "nichts passiert" |
| `count_reps(xs)` | **Liste von Winkeln**, ein Winkel pro Videobild, z.B. `[170, 90, 165, 85, 170]` | Anzahl Wiederholungen (`int`), hier `2` | `step` über die ganze Liste anwenden, beginnend bei `START`, am Schluss `.count` |

Die Winkel-Liste kommt aus dem gegebenen `angles` (pro Videobild ein Kniewinkel oder `None`), `None` entfernt das gegebene `drop_missing`.

![Zustandsautomat up/down mit den Schwellen 100° und 160°](docs/img/zustandsautomat.svg)

`step` hat **zwei Schwellen** (Hysterese): von `up` nach `down` unter `DOWN_BELOW` (100°), von `down` nach `up` über `UP_ABOVE` (160°), dabei zählt `count` eins hoch.

**Hinweis:**

```python
step(RepState("up", 0), 90)        # RepState("down", 0)   (gegeben)
step(RepState("down", 0), 170)     # RepState("up", 1)
step(RepState("up", 0), 130)       # RepState("up", 0)     (nichts passiert)
# Ein Fold: reduce(f, [a, b, c], start) == f(f(f(start, a), b), c)
```

**Prüfen:**

```bash
uv run pytest tests/test_reps.py -q
uv run python run_squats.py videos/kniebeugen.mp4 --plot
```

`run_squats.py` zählt Wiederholungen und speichert mit `--plot` den Winkelverlauf in `out/angles.png` (`--side right` für das rechte Bein). Schaut euch den Plot an: Passen die Schwellen 100° und 160° zu euren Kniebeugen?

> **Typisches Problem (unser Testvideo):** Etwa 10 Kniebeugen, gezählt wurden nur **3** (links) bzw. **4** (rechts) mit 100°/160° und `ema` 0.5. Gründe: Die Füsse waren abgeschnitten, also waren die Fussgelenke unsicher und bei vielen Bildern fehlte der Winkel (612 von 1023). Beim Aufstehen erreichte das Knie nur 140 bis 155°, also nie die 160°. Mit `UP_ABOVE = 140` kamen **9** bzw. **8** heraus.
>
> **Lehre:** Passt `DOWN_BELOW` und `UP_ABOVE` in `pipeline/reps.py` an eure Kurve an. Bei vielen Lücken weniger glätten (`alpha` 0.8 bis 1.0) oder nochmals filmen.
>
> **Werkzeug zum Prüfen:** Mit dem gegebenen `annotate_video.py` seht ihr, wo das Zählen schiefgeht:
>
> ```bash
> uv run python annotate_video.py videos/kniebeugen.mp4 --up 140
> ```
>
> Es schreibt `out/video/annotated.mp4`: Skelett, Kniewinkel und Zähler in jedem Bild. Spielt es z.B. mit VLC ab. Weitere Optionen: `--side left|right`, `--down`, `--up`, `--out`.

**Dauer:** Rechnet mit etwa 10 Bildern pro Sekunde auf einem Laptop (nur CPU). Ein Video von 30 s mit 30 Bildern/s (900 Bilder) braucht etwa 1.5 Minuten.

#### 4b: Eure eigene Analyse

**Das ist der letzte Teil: erst nach Runde 2, so weit ihr in der Zeit kommt.** Ihr wählt selbst, was ihr untersuchen wollt. Es gibt keine richtige Lösung: Wichtig ist, dass ihr eine Frage stellt, sie mit der Pipeline beantwortet und im Bericht beschreibt (optionaler Abschnitt am Schluss), was funktioniert hat und was nicht.

**Ideen:**

| Thema | Mögliche Frage | Was man misst |
|---|---|---|
| Fussball, Hockey, Rugby | Wie tief geht ein Spieler in die Knie? Wie oft wird gesprintet? | Kniewinkel über Zeit, Position der Hüfte von Bild zu Bild |
| Eishockey, Skaten | Wie stark ist die Beugung in der Grundstellung? Wie gleichmässig sind die Schritte? | Kniewinkel links und rechts, Rhythmus der Schritte |
| Kampfsport | Wie viele Kicks? Wie hoch? Wie gestreckt ist das Bein? | Fussgelenk höher als Hüfte (`y` kleiner, `y` zeigt nach unten!), maximaler Kniewinkel pro Kick |
| Krafttraining, Fitness | Liegestütze, Hampelmann, Ausfallschritte zählen | Ellbogen- oder Kniewinkel mit `step`/`count_reps` und eigenen Schwellen |
| Wurf, Schuss, Schlag | Wie ist der Arm beim Abwurf gestreckt? | Ellbogenwinkel (Schulter, Ellbogen, Handgelenk: 6, 8, 10 bzw. 5, 7, 9) |
| Links/rechts | Bewegt sich eine Seite anders als die andere? | dieselbe Messung für beide Seiten, Differenz |

**Bausteine:** Ihr habt alles schon. Für andere Gelenke ruft ihr `angle_at` mit den passenden drei Punkten auf, für mehrere Personen nehmt ihr alle aus `predict_image` statt nur die erste:

```python
from PIL import Image
from pipeline.reps import frames
from pipeline.inference import predict_image, confident
from pipeline.geometry import to_pixels, angle_at

def elbow_angles(path, side_idx=(6, 8, 10)):          # lazy: ein Winkel pro Videobild
    for frame in frames(path):
        h, w = frame.shape[:2]
        persons = predict_image(Image.fromarray(frame))
        if not persons:
            yield None
            continue
        p = to_pixels(confident(persons[0]), w, h)
        a, b, c = (p.keypoints[i] for i in side_idx)
        yield angle_at(a, b, c) if min(a.v, b.v, c.v) > 0 else None
```

Danach wie in 4a: `drop_missing`, `ema`, eigene Schwellen und `count_reps`, oder einfach `max(...)`, Mittelwert, ein Plot.

**Gut zu wissen bei echten Sportvideos:**

- **Mehrere Personen:** `predict_image` liefert alle erkannten Personen, sortiert nach Konfidenz, aber **ohne Identität**: Person 0 im einen Bild ist nicht unbedingt Person 0 im nächsten. Für eine bestimmte Person über die Zeit gibt es in Ultralytics `get_model(WEIGHTS).track(bild, persist=True)`: jede Person bekommt eine Nummer (`r.boxes.id`). Einfacher: ein Ausschnitt, in dem nur eine Person gut sichtbar ist.
- **Kleine Personen** (Spieler weit weg) werden schlechter erkannt, Punkte werden unsicher. Nahaufnahmen funktionieren besser.
- **Kamera:** Eine bewegte Kamera ist für Winkel egal, für Positionen (Sprint, Weg) nicht.
- **Rechenzeit:** Auf der CPU schafft das Modell einige Bilder pro Sekunde. Nehmt kurze Clips (10 bis 30 Sekunden) oder wertet nur jedes zweite oder dritte Bild aus.
- **Videos:** Am besten eigene Aufnahmen. Fremde Videos nur, wenn ihr sie verwenden dürft. Filmt niemanden ohne Einverständnis.

**Ergebnis:** euer Skript (z.B. `analyse.py`), ein Plot oder eine Tabelle in `out/` und ein Abschnitt im Bericht: Frage, Vorgehen, Ergebnis.

---

## 3. Runde 2: FP-Prinzipien

Euer Code läuft. Jetzt schauen wir, **warum er so gebaut ist**.

### 3.0 FP ist überall

| Sprache / Framework | Typische FP-Prinzipien |
|---|---|
| **Python** | `map`/`filter`, Comprehensions, `functools.reduce`/`partial`/`lru_cache`, `itertools`, `@dataclass(frozen=True)`, Generatoren |
| **React (+ Redux)** | Komponenten sind reine Funktionen ihrer Props; State wird nie verändert, sondern ersetzt; Reducer `(state, action) -> state` (wie `step` in Schritt 4) |
| **Apache Spark / PySpark** | unveränderliche Datasets, `map`/`filter`/`reduce`, lazy Auswertung (verwandt mit Hugging Face `datasets`) |
| **Java** | Streams, Lambdas, Records |
| **C#** | LINQ (`Select`, `Where`, `Aggregate`) |
| **JavaScript / TypeScript** | `map`/`filter`/`reduce`, Spread-Kopien (`{...obj}`, `[...arr]`) |
| **SQL** | deklarativ: Ergebnis beschreiben, nicht Schritte |
| **Kotlin / Scala / F#** | FP als Teil der Sprache: Immutability, Pattern Matching, Higher-Order Functions |

### Recherche im Betrieb (jede Person, ca. 20 min)

Fragt Kolleginnen und Kollegen, schaut in Projekte oder Dokumentationen eures Lehrbetriebs:

- Welche **Sprachen** und **Frameworks** verwendet euer Betrieb?
- Welches **Software-Design** steckt dahinter (z.B. Web-App, Microservices, Datenpipeline, Event-getrieben)?
- Welche **FP-Prinzipien** sind dort am prominentesten, und **wo zeigen sie sich**?

Eure Antworten kommen in **Teil B** des Berichts (ein Block pro Person). Dort beschreibt ihr auch, wo ein FP-Prinzip fehlt und welches Problem dadurch entsteht, ähnlich wie in den `bad/`-Skripten unten.

**Beispiel-Antwort (React):**
> Unser Betrieb baut Web-Oberflächen mit React und Redux. Eine Komponente ist eine Funktion, die aus Props die Anzeige berechnet, ohne etwas zu verändern: das ist eine reine Funktion. Der State wird nie direkt verändert, sondern im Reducer `(state, action) -> state` durch einen neuen Wert ersetzt. Das macht Fehler nachvollziehbar, weil man jede Aktion einzeln abspielen kann. Das ist dasselbe Muster wie `step` in unserer Kniebeugen-Zählung.

Wenn ihr in eurem Betrieb wenig Offensichtliches findet: Das ist selbst eine Aussage. Schaut, welche Prinzipien trotzdem versteckt vorkommen (z.B. `map`/`filter` im Code, SQL-Abfragen, Streams).

### Ablauf der Abschnitte 3.1 bis 3.4

Pro Abschnitt:

1. **Prinzip**: kurz erklärt, mit Stellen in eurem Code.
2. **Was bringt es?**: der Nutzen in einem Satz.
3. **Kaputt machen**: Das Skript in `bad/` ist **fertig**, ihr startet es nur.
4. **Beobachtet**: 2 bis 3 Fragen. Sie führen zu Frage 3 von Teil A im Bericht: *Was geht schief ohne dieses Prinzip?* Antwortet in euren eigenen Worten.
5. **Reparieren**: nur bei 3.3.

Die `bad/`-Skripte brauchen eure Lösungen aus Runde 1. Startet immer **vom Projektordner** mit `-m` (kein `/` im Namen, kein `.py`).

### 3.1 Immutability und Aliasing

**Prinzip:** Immutability (Unveränderlichkeit) heisst: Ein Wert wird nach dem Bauen nie mehr geändert, man baut stattdessen einen neuen. Bei euch ist das `tuple(Keypoint(...) for ...)` in `parse_label_line` und `to_pixels`: Ein Tupel aus `NamedTuple`s lässt sich nicht verändern. Veränderbare Daten (Listen, Dicts) haben ein Problem: Zwei Namen können auf **dasselbe** Objekt zeigen (Aliasing), und eine Änderung über den einen Namen sieht man am anderen. In Haskell ist alles unveränderlich.

**Was bringt es?** Niemand kann Daten hinter eurem Rücken ändern, darum könnt ihr Werte gefahrlos weitergeben und teilen.

**Kaputt machen:**

```bash
uv run python -m bad.b1_aliasing
```

**Beobachtet:**

1. Warum ist `backup` auch verändert, obwohl ihr es nie angefasst habt?
2. Warum bleibt `p` in Teil 3 unverändert? Was passiert in Teil 2, und warum?
3. Was geht ohne Immutability schief, und wo könnte das in eurem Betrieb passieren (Teil A, Frage 3; Teil B)?

### 3.2 Pure functions

**Prinzip:** Eine pure function (reine Funktion) liefert bei gleichen Argumenten immer dasselbe Ergebnis und hat keine side effects (Nebenwirkungen: verändert nichts ausserhalb). Das seht ihr an `return Person(box=box, keypoints=kps, score=p.score)` in `to_pixels`: Die Funktion baut eine **neue** Person, die alte bleibt. Man nennt das referenzielle Transparenz: Man könnte den Aufruf durch sein Ergebnis ersetzen. Auch `angle_at` und `knee_angle` sind pure.

**Was bringt es?** Pure Funktionen kann man einzeln testen, in beliebiger Reihenfolge aufrufen und beliebig oft wiederholen.

**Kaputt machen:**

```bash
uv run python -m bad.b2_pixels_inplace
```

Das Skript hat eine `to_pixels_inplace`-Variante, die die Koordinaten in der Datenstruktur überschreibt.

**Beobachtet:**

1. Wie ändern sich Knie-Koordinaten und Winkel beim zweiten Aufruf auf demselben Objekt?
2. Was passiert, wenn man die Reihenfolge "Winkel berechnen / Skelett zeichnen" vertauscht? Wäre das bei eurer reinen `to_pixels` auch so?
3. Wie stark weicht der Winkel nach der Doppel-Skalierung ab? Warum ändert sich ein Winkel nicht, wenn man x und y mit **demselben** Faktor skaliert, hier aber schon?

### 3.3 Functional core, imperative shell und Memoization

**Prinzip:** Programme brauchen side effects (Dateien laden, Modell starten), aber man hält sie in einer dünnen unreinen Schale (imperative shell) und den Rest in einem reinen Kern (functional core). `result_to_persons` ist der Kern: nur Zahlen rein, Personen raus, einfach zu testen. `get_model`, `predict_batch` und `predict_image` sind die Schale. Memoization (Zwischenspeichern) mit `@lru_cache` auf `get_model` ist nur bei Funktionen erlaubt, deren Ergebnis **vollständig** von den Argumenten abhängt: Der Cache merkt sich das Ergebnis pro Argumente. Hängt es von mehr ab (z.B. einer globalen Variable), liefert der Cache veraltete Werte. In Haskell macht der Typ (`IO`) die Unreinheit sichtbar.

**Was bringt es?** Der reine Kern ist ohne Modell und ohne Dateien testbar, und Memoization spart Rechenzeit, solange die Funktion rein ist.

**Kaputt machen:**

```bash
uv run python -m bad.b4_stale_cache
```

Eine Funktion `count_people` zählt Personen auf Bild 110, liest dabei aber ein globales `CONF` und ist mit `lru_cache` memoisiert. Das Skript ändert `CONF` und ruft nochmals auf.

**Beobachtet:**

1. Welche zwei Zahlen druckt das Skript für `count_people`, welche erwartet ihr nach `CONF = 0.9`?
2. Wieso rechnet die zweite Ausführung nicht neu? Was ist der Schlüssel des Caches?
3. Wieso ist `count_people` "nicht rein"? Was druckt `fixed`?

**Reparieren:** Das Skript enthält `count_people_fixed(path, conf)`. Schreibt `count_people` selber so um, dass `conf` ein **Argument** ist, und erklärt, warum der Cache jetzt stimmt.

**Optional:** Das Dataset-`map` hat auch einen Cache (Dateien `cache-*.arrow` unter `data/coco8-pose-ds/`, gilt nur für Datasets, die von der Festplatte mit `load_from_disk` geladen wurden). Was passiert, wenn `predict_batch` ein globales `CONF` aus einem **importierten Modul** liest statt `conf` als Argument? Ändert `CONF`, startet erneut und schaut, ob neu gerechnet wird. `predict_split` gibt `conf` darum über `fn_kwargs` weiter.

### 3.4 State as a value und Fold

**Prinzip:** Wo Zustand nötig ist (Wiederholungszähler), packt man ihn in einen **Wert** (`RepState`, state as a value) und schreibt eine pure function `step(state, angle) -> neuer state`. Das ist ein **Reducer** wie in Redux. `reduce(step, angles, START)` in `count_reps` ist dann ein fold (in Haskell `foldl`): Die Liste wird Winkel für Winkel zu einem einzigen Wert gefaltet. Es gibt keinen versteckten globalen Zähler: gleiche Eingabe, gleiches Ergebnis. `states` zeigt dasselbe als Scan (alle Zwischenzustände), `frames` ist ein Generator (lazy: ein Bild nach dem anderen im Speicher).

**Was bringt es?** Der Zustand ist sichtbar und gehört zum Aufruf, darum kann man jeden Schritt einzeln testen, wiederholen und im Diagramm verfolgen.

**Kaputt machen:**

```bash
uv run python -m bad.b6_reps
```

Das Skript vergleicht einen **globalen** Zähler mit eurem reinen `count_reps`.

**Beobachtet:**

1. Was gibt `count_reps_global` beim 1. und beim 2. Aufruf auf derselben Eingabe zurück, was euer `count_reps`?
2. Welche Information steckt bei der globalen Version in `count` und `phase`, und wo steckt sie bei `RepState`?
3. Wie würdet ihr die globale Version testen? Und eure? Was geht ohne state as a value schief?

---

## 4. Bericht und Abgabe

Ihr füllt die Vorlage **`bericht.md`** im Repo aus, am besten laufend. Sie hat drei Teile: Teil A (Code und FP-Prinzipien, pro Schritt drei kurze Antworten), Teil B (FP im Lehrbetrieb, ein Block pro Person) und Teil C (Fazit, KI). Optional am Schluss: Schritt 4b.

### Abgabe

Euer **privates** Repo (alle Teammitglieder und die Lehrperson `tinunit` als Collaborators) mit:

- [ ] Tests für die Schritte 1 bis 4 grün (`uv run pytest -q`)
- [ ] `uv run python run_squats.py videos/kniebeugen.mp4` läuft (die Videos selbst nicht committen)
- [ ] `bericht.md` ausgefüllt
- [ ] ein Commit pro Schritt, mit aussagekräftiger Nachricht

### Bewertung

Bewertet wird der **Bericht: 20 Punkte**. Der Code (Tests grün, `run_squats.py` läuft) ist eine Voraussetzung und gibt keine Punkte.

| Teil | Punkte |
|---|---|
| A: Code und FP-Prinzipien (4 Schritte × 3 Punkte) | 12 |
| B: FP im Lehrbetrieb (pro Person, Teampunktzahl = Durchschnitt) | 6 |
| C: Fazit und KI | 2 |
| **Total** | **20** |

---

## 5. Spickzettel

| Idee | Begriff (EN) | Python | Haskell |
|---|---|---|---|
| Liste umwandeln | map | `[f(x) for x in xs]`, `map(f, xs)` | `map f xs` |
| Liste filtern | filter | `[x for x in xs if p(x)]`, `filter(p, xs)` | `filter p xs` |
| Zusammenfalten | fold | `reduce(f, xs, start)` | `foldl f start xs` |
| Zwischenergebnisse | scan | `accumulate(xs, f, initial=s)` | `scanl f s xs` |
| Funktion als Wert | higher-order function | `def g(n): return lambda x: ...` | `g n = \x -> ...` |
| Unveränderlich | immutability | `NamedTuple`, Tupel, `frozen=True` | alles |
| Kopie mit Änderung | copy with change | `p._replace(...)`, `{**d, "k": v}` | `p { field = v }` |
| Keine Nebenwirkung | pure function | gleiche Eingabe, gleiche Ausgabe | alles ausser `IO` |
| Fehlender Wert | optional | `Optional[X]` (`None`) | `Maybe X` |
| Lazy Folge | generator | Generator mit `yield` | normale Liste (lazy) |
| Zwischenspeichern | memoization | `@lru_cache` (nur bei reinen Funktionen!) | nicht nötig (rein) |
| Zustand als Wert | reducer | `step(state, x) -> state` | `foldl step start xs` |

### Befehle

| Was | Befehl |
|---|---|
| Alle Tests | `uv run pytest -q` |
| Tests eines Schritts | `uv run pytest tests/test_<x>.py -q` |
| Label und Vorhersage als Bild | `uv run python visualize.py` (Ausgabe `out/`) |
| Eigene Fotos | `uv run python run_own.py` (Fotos in `eigene/`, Ausgabe `out/eigene/`) |
| Video auswerten | `uv run python run_squats.py videos/kniebeugen.mp4 --plot` |
| Video mit Skelett, Winkel, Zähler | `uv run python annotate_video.py videos/kniebeugen.mp4 --up 140` (Ausgabe `out/video/annotated.mp4`) |
| Fehler provozieren | `uv run python -m bad.<name>` (`b1_aliasing`, `b2_pixels_inplace`, `b4_stale_cache`, `b6_reps`) |
