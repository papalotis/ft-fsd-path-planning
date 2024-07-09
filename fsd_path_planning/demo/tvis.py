import json
from pathlib import Path

import matplotlib.pyplot as plt

x = json.loads((Path(__file__).parent / "accel2.json").read_text())


possx = []
possy = []

for d in x:
    px, py = d["car_position"]

    possx.append(px)
    possy.append(py)

plt.plot(possx, possy)

plt.axis("equal")

plt.show()
