# Modul 323: Pose-Pipeline (Starter-Repo)

Ihr baut eine Pipeline, die auf Bildern Personen erkennt, Gelenkpunkte (Keypoints) bestimmt, mit Labels vergleicht und Bewegungen in einem Video auswertet, funktional in Python. Zuerst implementieren (Runde 1), dann FP-Prinzipien im eigenen Code entdecken (Runde 2).

**Einstieg: [ANLEITUNG.md](ANLEITUNG.md).** Dort steht alles, vom Setup bis zur Abgabe. Euren Bericht schreibt ihr in `bericht.md`.

## Setup

### Voraussetzungen

- Pro Person ein **GitHub-Konto**.
- **git** und **uv** installiert. uv installieren:
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Repo einrichten (einmal pro Team)

1. **Eine Person** des Teams öffnet dieses Repo auf GitHub und klickt **"Use this template" → "Create a new repository"**.
2. Name z.B. `m323-pose-<eure-namen>`, Sichtbarkeit **Private**, dann **Create repository**.
3. Im neuen Repo: **Settings → Collaborators → Add people**:
   - **alle anderen Teammitglieder** (Teams: 2 bis 4 Personen),
   - die **Lehrperson** (GitHub: `tinunit`).
4. Alle klonen das neue Repo und richten es ein:

   ```bash
   git clone git@github.com:<euer-konto>/<euer-repo>.git
   cd <euer-repo>
   uv sync
   ```

**Nicht forken:** Ein Fork eines öffentlichen Repos ist selbst öffentlich, dann sieht die ganze Klasse euren Code. Mit "Use this template" bekommt ihr ein eigenes, privates Repo.

**Zusammenarbeiten:** Vor dem Arbeiten `git pull`, danach `git add`, `git commit`, `git push`. Arbeitet in verschiedenen Dateien. Ein Commit pro Schritt, z.B. `git commit -m "Schritt 2: Geometrie"`.

## Tests und Skripte

```bash
uv run pytest -q                          # alle Tests (anfangs sind die Tests eurer Stubs rot)
uv run pytest tests/test_data.py -q       # Tests eines Schritts
uv run python run_pipeline.py             # Gesamtlauf
uv run python visualize.py                # Label und Vorhersage als Bild (out/)
uv run python run_own.py                  # eigene Fotos aus eigene/ (Ausgabe out/eigene/)
uv run python run_squats.py videos/kniebeugen.mp4 --plot
uv run python -m bad.b1_aliasing     # Runde 2: Fehler provozieren
```

Startet alles im Projektordner.

## Arbeitsweise

Bericht (`bericht.md`) laufend ausfüllen, ein Commit pro Schritt. Eigene Fotos (`eigene/`) und Videos werden **nicht** committed.
