from pathlib import Path

import numpy as np
from trimesh.points import PointCloud

from .anvil import Block, EmptyRegion
from .anvil.errors import OutOfBoundsCoordinates


class Minecraft:
    def __init__(self, point_cloud: PointCloud) -> None:
        self.point_cloud = point_cloud

    def _point_shift(self, points: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
        points += np.array([x, y, z])
        return points

    def _split_point_cloud(self, vertices: np.ndarray, block_size: int = 512) -> dict[str, np.ndarray]:
        block_coords = np.floor_divide(vertices[:, :2], block_size).astype(int)
        unique, inverse = np.unique(block_coords, axis=0, return_inverse=True)

        blocks = {}
        for idx, (bx, by) in enumerate(unique):
            mask = inverse == idx
            block_id = f"r.{bx}.{by}.mca"
            blocks[block_id] = vertices[mask]

        return blocks

    def _standardize_vertices(self, blocks: dict[str, np.ndarray], block_size: int = 512):
        return {block_id: np.mod(vertices, block_size) for block_id, vertices in blocks.items()}

    def build_region(self, output: Path, origin: tuple[float, float, float] | None = None) -> None:
        points = np.asarray(self.point_cloud.vertices)

        origin_point = self._get_world_origin(points) if origin is None else origin
        print(f"origin_point: {origin_point}")

        # 点群の中心を原点に移動
        points = self._point_shift(points, -origin_point[0], -origin_point[1], 0)
        # ボクセル中心を原点とする。ボクセルは1m間隔なので、原点を右に0.5m、下に0.5mずらす
        points = self._point_shift(points, 0.5, 0.5, 0)
        # Y軸を反転させて、Minecraftの南北とあわせる
        points[:, 1] *= -1

        # 原点を中心として、x軸方向に512m、y軸方向に512mの領域を作成する
        # 領域ごとに、ボクセルの点群を分割する
        # 分割した点群を、領域ごとに保存する
        blocks = self._split_point_cloud(points)
        standardized_blocks = self._standardize_vertices(blocks)

        stone = Block("minecraft", "stone")

        # Delete contents under ``output/world_data/region`` if it exists
        # or create the directory otherwise
        region_dir = Path(output) / "world_data" / "region"
        if region_dir.exists():
            for file in region_dir.iterdir():
                file.unlink()
        else:
            region_dir.mkdir(parents=True, exist_ok=True)

        for block_id, points in standardized_blocks.items():
            region = EmptyRegion(0, 0)
            points = np.asarray(points).astype(int)
            for row in points:
                x, y, z = row
                try:
                    region.set_block(stone, x, z, y)  # MinecraftとはY-UPの右手系なので、そのように変数を定義する
                except OutOfBoundsCoordinates:
                    continue
            print(f"save: {block_id}")
            region.save(str(region_dir / block_id))

    def _get_world_origin(self, vertices):
        min_x = min(vertices[:, 0])
        max_x = max(vertices[:, 0])

        min_y = min(vertices[:, 1])
        max_y = max(vertices[:, 1])

        # 中心座標を求める
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2

        # 中心座標を右に0.5m、下に0.5mずらす
        origin_point = (center_x + 0.5, center_y + 0.5)

        return origin_point
