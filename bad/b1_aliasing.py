# Runde 2 – Schritt 1: Immutability (Unveränderlichkeit) und Aliasing.
# Startet mit `uv run python -m bad.b1_aliasing`.
# Beobachtet:
#   - Warum ist `backup` auch verändert, obwohl ihr es nie angefasst habt?
#   - Warum bleibt `p` in Teil 3 unverändert?
#   - Wo könnten in eurem Betrieb zwei Namen unbemerkt auf dasselbe Objekt zeigen?
from pipeline.data import parse_label_line
from pipeline.types import Keypoint

line = open("data/coco8-pose/labels/train/000000000036.txt").readline()

print("1. Mit Listen: zwei Namen, ein Objekt")
kps = [list(k) for k in parse_label_line(line).keypoints]   # veränderbare Kopie als Listen
backup = kps                                                 # "Sicherung"
kps[0][0] = 999                                              # Nase verschieben
print("   backup, Nase:", backup[0])

print("2. Mit NamedTuple: ändern geht nicht")
p = parse_label_line(line)
try:
    p.keypoints[0].x = 999
except AttributeError as e:
    print("   Fehler:", e)

print("3. Der funktionale Weg: neue Person bauen, alte bleibt")
nose = p.keypoints[0]._replace(x=999)
p2 = p._replace(keypoints=(nose,) + p.keypoints[1:])
print("   alt:", p.keypoints[0].x, " neu:", p2.keypoints[0].x)
