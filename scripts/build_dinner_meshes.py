"""Build small, original, meter-scale STL props without external CAD dependencies.

The closed radial profiles include inner walls and a base: they render hollow,
but must remain visual-only because MuJoCo mesh collision uses a convex hull.
Regenerate with: python scripts/build_dinner_meshes.py
"""

from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np


def write_lathe(path: Path, profile: list[tuple[float, float]], segments: int = 64) -> None:
    """Revolve a closed (radius, height) profile around the z axis."""
    rings = np.asarray([
        [(r * math.cos(a), r * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)]
        for r, z in profile
    ])
    triangles = []
    for i in range(len(profile)):
        for j in range(segments):
            a, b = rings[i, j], rings[i, (j + 1) % segments]
            c, d = rings[(i + 1) % len(profile), j], rings[(i + 1) % len(profile), (j + 1) % segments]
            for vertices in ((a, b, c), (b, d, c)):
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                length = np.linalg.norm(normal)
                if length > 1e-12:
                    triangles.append((normal / length, *vertices))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"Original bimanual dinner prop; meters".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            stream.write(struct.pack("<12fH", *np.asarray(triangle).ravel(), 0))
    print(f"{path}: {len(triangles)} triangles")


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "assets/robots/aloha/assets"
    write_lathe(output / "dinner_tumbler.stl", [
        (0, 0), (.021, 0), (.023, .003), (.025, .087), (.024, .090),
        (.022, .090), (.021, .087), (.019, .007), (0, .007),
    ])
    write_lathe(output / "dinner_vase.stl", [
        (0, 0), (.025, 0), (.030, .008), (.031, .045), (.026, .080),
        (.018, .105), (.018, .110), (.015, .110), (.015, .103),
        (.023, .078), (.028, .045), (.027, .012), (0, .008),
    ])


if __name__ == "__main__":
    main()
