import numpy as np

from plateau2minecraft.types import TriangleMesh


def align_base_height(mesh: TriangleMesh, target_height: float = 0.0) -> TriangleMesh:
    """Shift mesh so the lowest Z becomes ``target_height``."""
    vertices = np.asarray(mesh.vertices)
    offset = target_height - vertices[:, 2].min()
    vertices[:, 2] += offset
    return TriangleMesh(list(vertices), mesh.triangles)
