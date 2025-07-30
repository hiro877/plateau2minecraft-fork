import argparse
import logging
from pathlib import Path

from plateau2minecraft.converter import Minecraft
from plateau2minecraft.impart_color import assign
from plateau2minecraft.merge_points import merge
from plateau2minecraft.parser import get_triangle_meshs
from plateau2minecraft.voxelizer import voxelize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _extract_feature_type(file_path: Path) -> str:
    return file_path.stem.split("_")[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        nargs="*",
        help="the output result encompasses the specified CityGML range",
    )
    parser.add_argument("--output", required=True, type=Path, help="output folder")
    parser.add_argument(
        "--base-z",
        type=float,
        default=0.0,
        help="align all objects so that their lowest Z matches the specified height (default: 0)",
    )
    parser.add_argument(
        "--print-min-z",
        action="store_true",
        help="print the minimum Z of the processed data",
    )
    args = parser.parse_args()

    point_cloud_list = []
    for file_path in args.target:
        logging.info(f"Processing start: {file_path}")
        feature_type = _extract_feature_type(file_path)

        logging.info(f"Triangulation: {file_path}")
        triangle_mesh = get_triangle_meshs(file_path, feature_type)

        logging.info(f"Voxelize: {file_path}")
        point_cloud = voxelize(triangle_mesh)
        point_cloud = assign(point_cloud, feature_type)

        current_min_z = point_cloud.vertices[:, 2].min()
        offset = args.base_z - current_min_z
        point_cloud.vertices[:, 2] += offset

        point_cloud_list.append(point_cloud)
        logging.info(f"Processing end: {file_path}")

    logging.info(f"Merging: {args.target}")
    merged = merge(point_cloud_list)

    if args.print_min_z:
        min_z = merged.vertices[:, 2].min()
        print(f"minimum_z: {min_z}")

    logging.info(f"To : {args.target}")
    region = Minecraft(merged).build_region(args.output)
