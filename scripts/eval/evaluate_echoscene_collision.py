"""Evaluate EchoScene object meshes with PhyScene-style collision metrics.

EchoScene exports final object meshes, while ``calc_ckl.py`` reconstructs
meshes from PhyScene bbox/object-feature predictions. This script bridges that
gap by reading EchoScene's per-scene OBJ files directly and applying the same
collision-rate convention used by PhyScene:

* ColObj: fraction of objects that collide with at least one other object.
* ColScene: fraction of scenes that contain at least one collision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhyScene-style collision metrics on EchoScene OBJ outputs."
    )
    parser.add_argument(
        "--object-mesh-root",
        default="2050/echoscene/object_meshes",
        help="Directory containing one subdirectory per scene with object OBJ files.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="Scene id to evaluate. Can be passed multiple times. Defaults to all scenes.",
    )
    parser.add_argument(
        "--scene-json",
        default=None,
        help=(
            "Optional EchoScene/PhyScene-style JSON. Scene ids inside it restrict "
            "scenes, and it can provide object transforms."
        ),
    )
    parser.add_argument(
        "--transform-source",
        choices=("none", "json"),
        default="none",
        help=(
            "Use 'json' when OBJ files are local/normalized object meshes and "
            "the scene JSON contains translations, sizes, and angles."
        ),
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for JSON sizes when transforming local OBJ meshes. Use "
            "1.0 if sizes are full extents, 2.0 if sizes are half extents."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("mesh", "bbox_no_direction"),
        default="mesh",
        help="Collision method. 'mesh' mirrors PhyScene's mesh/point-in-mesh check.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=100,
        help="Maximum number of scenes to evaluate, matching calc_ckl.py's default cap.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for mesh collision.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "kaolin", "ray"),
        default="auto",
        help="Mesh inside-test backend. 'auto' tries Kaolin, then a torch ray test.",
    )
    parser.add_argument(
        "--point-chunk-size",
        type=int,
        default=2048,
        help="Number of query points per chunk for the ray backend.",
    )
    parser.add_argument(
        "--face-chunk-size",
        type=int,
        default=8192,
        help="Number of triangles per chunk for the ray backend.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a machine-readable metric summary.",
    )
    parser.add_argument(
        "--debug-sanity",
        action="store_true",
        help=(
            "Print deterministic sanity logs for transform axes, OBJ-to-JSON "
            "indices, transformed bounds, and collision pairs."
        ),
    )
    parser.add_argument(
        "--debug-transform",
        action="store_true",
        help="Print local, target, and transformed extents for each transformed OBJ.",
    )
    parser.add_argument(
        "--debug-index",
        action="store_true",
        help="Print filename-derived object index and JSON class/size for each OBJ.",
    )
    parser.add_argument(
        "--debug-pairs",
        action="store_true",
        help="Print bbox-overlapping pairs and mesh-colliding pairs.",
    )
    parser.add_argument(
        "--run-self-tests",
        action="store_true",
        help="Run deterministic cube collision checks before evaluating scenes.",
    )
    return parser.parse_args()


def load_scene_payload(path: str | None) -> dict | None:
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_ids_from_payload(payload: dict | None) -> list[str] | None:
    if payload is None:
        return None
    scene_ids = payload.get("scene_ids")
    if scene_ids is None:
        raise ValueError("Scene JSON does not contain a 'scene_ids' field")

    return [str(scene_id) for scene_id in scene_ids]


def scene_payload_by_id(payload: dict | None) -> dict[str, dict]:
    if payload is None:
        return {}

    scene_ids = payload.get("scene_ids")
    if scene_ids is None:
        raise ValueError("Scene JSON does not contain a 'scene_ids' field")

    mapping = {}
    for scene_index, scene_id in enumerate(scene_ids):
        mapping[str(scene_id)] = {
            "class_labels": np.asarray(payload["class_labels"][scene_index])
            if "class_labels" in payload
            else None,
            "translations": np.asarray(payload["translations"][scene_index]),
            "sizes": np.asarray(payload["sizes"][scene_index]),
            "angles": np.asarray(payload["angles"][scene_index]),
            "objectness": np.asarray(payload["objectness"][scene_index])
            if "objectness" in payload
            else None,
        }
    return mapping


def find_scene_dirs(root: Path, requested_scene_ids: Iterable[str] | None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Object mesh root does not exist: {root}")

    if requested_scene_ids is not None:
        scene_dirs = [root / scene_id for scene_id in requested_scene_ids]
    else:
        scene_dirs = [p for p in root.iterdir() if p.is_dir()]

    missing = [p for p in scene_dirs if not p.exists()]
    if missing:
        missing_text = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Missing scene mesh directories:\n{missing_text}")

    return sorted(scene_dirs)


class ObjMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces
        self.bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)


def parse_obj_index(token: str, vertex_count: int) -> int:
    index = int(token.split("/")[0])
    if index < 0:
        return vertex_count + index
    return index - 1


def load_mesh(path: Path) -> ObjMesh:
    vertices = []
    faces = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                indices = [parse_obj_index(token, len(vertices)) for token in tokens]
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")

    return ObjMesh(vertices, faces)


def object_index_from_filename(path: Path) -> int | None:
    stem_parts = path.stem.split("_")
    if not stem_parts:
        return None
    try:
        return int(stem_parts[-1]) - 1
    except ValueError:
        return None


def transform_mesh_from_json(
    mesh: ObjMesh,
    object_index: int,
    scene_payload: dict,
    size_scale: float,
    path: Path | None = None,
    debug_transform: bool = False,
) -> ObjMesh:
    translations = scene_payload["translations"]
    sizes = scene_payload["sizes"]
    angles = scene_payload["angles"]
    if object_index < 0 or object_index >= len(translations):
        raise IndexError(f"Object index {object_index + 1} is outside the scene JSON")

    vertices = mesh.vertices.copy()
    local_min = vertices.min(axis=0)
    local_max = vertices.max(axis=0)
    local_center = (local_min + local_max) / 2.0
    local_extent = local_max - local_min
    local_extent[local_extent == 0] = 1.0

    target_extent = np.asarray(sizes[object_index], dtype=np.float32) * size_scale
    vertices = (vertices - local_center) * (target_extent / local_extent)

    theta = float(np.asarray(angles[object_index]).reshape(-1)[0])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation = np.asarray(
        [
            [cos_theta, 0.0, -sin_theta],
            [0.0, 1.0, 0.0],
            [sin_theta, 0.0, cos_theta],
        ],
        dtype=np.float32,
    )
    vertices = vertices @ rotation.T
    vertices = vertices + np.asarray(translations[object_index], dtype=np.float32)
    transformed_mesh = ObjMesh(vertices.astype(np.float32), mesh.faces)

    if debug_transform:
        name = path.name if path is not None else f"object_{object_index + 1}"
        transformed_extent = transformed_mesh.bounds[1] - transformed_mesh.bounds[0]
        print("[transform]", name)
        print("  object_index =", object_index)
        print("  local_extent =", local_extent.tolist())
        print("  target_extent =", target_extent.tolist())
        print("  transformed_extent =", transformed_extent.tolist())
        print("  translation =", np.asarray(translations[object_index]).tolist())
        print("  angle =", theta)

    return transformed_mesh


def load_scene_meshes(
    scene_dir: Path,
    transform_source: str,
    scene_payload: dict | None,
    size_scale: float,
    debug_transform: bool,
    debug_index: bool,
) -> list[tuple[Path, ObjMesh]]:
    meshes = []
    paths = list(scene_dir.glob("*.obj"))
    if transform_source == "json":
        paths = sorted(
            paths,
            key=lambda p: (
                object_index_from_filename(p)
                if object_index_from_filename(p) is not None
                else 10**9,
                p.name,
            ),
        )
    else:
        paths = sorted(paths)

    for path in paths:
        mesh = load_mesh(path)
        if transform_source == "json":
            if scene_payload is None:
                raise ValueError("--transform-source json requires --scene-json")
            object_index = object_index_from_filename(path)
            if object_index is None:
                raise ValueError(f"Could not read object index from filename: {path.name}")
            class_labels = scene_payload.get("class_labels")
            class_index = None
            if class_labels is not None and object_index < len(class_labels):
                class_index = int(np.asarray(class_labels[object_index]).argmax())
            if debug_index:
                print("[index]", path.name)
                print("  filename_object_index =", object_index)
                print("  filename_category_id =", filename_category_id(path))
                print("  json_class_index =", class_index)
                print("  json_size =", np.asarray(scene_payload["sizes"][object_index]).tolist())
            objectness = scene_payload.get("objectness")
            if objectness is not None and objectness[object_index, 0] <= 0:
                if debug_index:
                    print("  skipped_objectness =", float(objectness[object_index, 0]))
                continue
            mesh = transform_mesh_from_json(
                mesh,
                object_index,
                scene_payload,
                size_scale,
                path=path,
                debug_transform=debug_transform,
            )
        meshes.append((path, mesh))
    if not meshes:
        raise ValueError(f"No OBJ files found in {scene_dir}")
    return meshes


def filename_category_id(path: Path) -> int | None:
    stem_parts = path.stem.split("_")
    if len(stem_parts) < 2:
        return None
    try:
        return int(stem_parts[-2])
    except ValueError:
        return None


def bounds_overlap(mesh_a: ObjMesh, mesh_b: ObjMesh) -> bool:
    box_a = np.asarray(mesh_a.bounds)
    box_b = np.asarray(mesh_b.bounds)
    if box_a[0, 0] >= box_b[1, 0] or box_b[0, 0] >= box_a[1, 0]:
        return False
    if box_a[0, 1] >= box_b[1, 1] or box_b[0, 1] >= box_a[1, 1]:
        return False
    if box_a[0, 2] >= box_b[1, 2] or box_b[0, 2] >= box_a[1, 2]:
        return False
    return True


def bbox_collision_flags(
    meshes: list[tuple[Path, ObjMesh]],
    debug_pairs: bool = False,
) -> np.ndarray:
    flags = np.zeros(len(meshes), dtype=bool)
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            if bounds_overlap(meshes[i][1], meshes[j][1]):
                flags[i] = True
                flags[j] = True
                if debug_pairs:
                    print("[bbox-pair]", meshes[i][0].name, "<->", meshes[j][0].name)
    return flags


def kaolin_check_any_inside(mesh_i: ObjMesh, mesh_j: ObjMesh, device: str) -> bool:
    import torch
    from kaolin.ops.mesh import check_sign

    verts_i = torch.as_tensor(
        np.asarray(mesh_i.vertices), dtype=torch.float32, device=device
    ).unsqueeze(0)
    faces_i = torch.as_tensor(np.asarray(mesh_i.faces), dtype=torch.long, device=device)
    points_j = torch.as_tensor(
        np.asarray(mesh_j.vertices), dtype=torch.float32, device=device
    ).unsqueeze(0)
    occupancy = check_sign(verts_i, faces_i, points_j)
    return bool(occupancy.max().item() > 0)


def ray_check_any_inside(
    mesh_i: ObjMesh,
    mesh_j: ObjMesh,
    device: str,
    point_chunk_size: int,
    face_chunk_size: int,
) -> bool:
    import torch

    eps = 1e-7
    vertices = torch.as_tensor(mesh_i.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh_i.faces, dtype=torch.long, device=device)
    points = torch.as_tensor(mesh_j.vertices, dtype=torch.float32, device=device)
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)

    for point_start in range(0, points.shape[0], point_chunk_size):
        point_chunk = points[point_start : point_start + point_chunk_size]
        hit_counts = torch.zeros(point_chunk.shape[0], dtype=torch.int32, device=device)

        for face_start in range(0, faces.shape[0], face_chunk_size):
            face_chunk = faces[face_start : face_start + face_chunk_size]
            triangles = vertices[face_chunk]
            v0 = triangles[:, 0, :]
            v1 = triangles[:, 1, :]
            v2 = triangles[:, 2, :]
            edge1 = v1 - v0
            edge2 = v2 - v0

            h = torch.cross(
                direction.expand(edge2.shape[0], 3), edge2, dim=1
            )
            a = (edge1 * h).sum(dim=1)
            valid_triangle = torch.abs(a) > eps
            if not bool(valid_triangle.any().item()):
                continue

            inv_a = torch.zeros_like(a)
            inv_a[valid_triangle] = 1.0 / a[valid_triangle]

            s = point_chunk[:, None, :] - v0[None, :, :]
            u = inv_a[None, :] * (s * h[None, :, :]).sum(dim=2)
            valid_u = (u >= 0.0) & (u <= 1.0) & valid_triangle[None, :]
            if not bool(valid_u.any().item()):
                continue

            q = torch.cross(s, edge1[None, :, :], dim=2)
            v = inv_a[None, :] * (
                q * direction.view(1, 1, 3)
            ).sum(dim=2)
            t = inv_a[None, :] * (edge2[None, :, :] * q).sum(dim=2)

            hits = valid_u & (v >= 0.0) & ((u + v) <= 1.0) & (t > eps)
            hit_counts += hits.sum(dim=1).to(torch.int32)

        if bool((hit_counts % 2 == 1).any().item()):
            return True

    return False


def mesh_collision_flags(
    meshes: list[tuple[Path, ObjMesh]],
    device: str,
    backend: str,
    point_chunk_size: int,
    face_chunk_size: int,
    debug_pairs: bool = False,
) -> np.ndarray:
    check_any_inside = None
    if backend in ("auto", "kaolin"):
        try:
            import kaolin  # noqa: F401

            check_any_inside = lambda a, b: kaolin_check_any_inside(a, b, device)
        except Exception as exc:
            if backend == "kaolin":
                raise
            print(f"Kaolin backend unavailable, falling back to ray backend: {exc}")

    if check_any_inside is None:
        check_any_inside = lambda a, b: ray_check_any_inside(
            a, b, device, point_chunk_size, face_chunk_size
        )

    flags = np.zeros(len(meshes), dtype=bool)
    for i, (_, mesh_i) in enumerate(meshes):
        for j in range(i + 1, len(meshes)):
            if flags[i] and flags[j]:
                continue
            _, mesh_j = meshes[j]
            if not bounds_overlap(mesh_i, mesh_j):
                continue

            if debug_pairs:
                print("[candidate-pair]", meshes[i][0].name, "<->", meshes[j][0].name)
            if check_any_inside(mesh_i, mesh_j):
                flags[i] = True
                flags[j] = True
                if debug_pairs:
                    print("[collision-pair]", meshes[i][0].name, "<->", meshes[j][0].name)

    return flags


def evaluate_scene(
    scene_dir: Path,
    method: str,
    device: str,
    backend: str,
    point_chunk_size: int,
    face_chunk_size: int,
    transform_source: str,
    scene_payload: dict | None,
    size_scale: float,
    debug_transform: bool,
    debug_index: bool,
    debug_pairs: bool,
) -> tuple[np.ndarray, list[tuple[Path, ObjMesh]]]:
    meshes = load_scene_meshes(
        scene_dir,
        transform_source,
        scene_payload,
        size_scale,
        debug_transform,
        debug_index,
    )
    if method == "bbox_no_direction":
        return bbox_collision_flags(meshes, debug_pairs), meshes
    return (
        mesh_collision_flags(
            meshes, device, backend, point_chunk_size, face_chunk_size, debug_pairs
        ),
        meshes,
    )


def make_cube(center: tuple[float, float, float], extent: float) -> ObjMesh:
    cx, cy, cz = center
    h = extent / 2.0
    vertices = np.asarray(
        [
            [cx - h, cy - h, cz - h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz + h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx - h, cy + h, cz + h],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return ObjMesh(vertices, faces)


def run_self_tests(device: str, point_chunk_size: int, face_chunk_size: int) -> None:
    print("[self-test] ray backend cube checks")
    cube_a = make_cube((0.0, 0.0, 0.0), 1.0)
    cube_overlap = make_cube((0.25, 0.0, 0.0), 0.5)
    cube_apart = make_cube((2.0, 0.0, 0.0), 0.5)
    overlap = ray_check_any_inside(
        cube_a, cube_overlap, device, point_chunk_size, face_chunk_size
    )
    separate = ray_check_any_inside(
        cube_a, cube_apart, device, point_chunk_size, face_chunk_size
    )
    print("  overlapping_cubes_expected=True actual=", overlap)
    print("  separated_cubes_expected=False actual=", separate)


def main() -> None:
    args = parse_args()
    root = Path(args.object_mesh_root)
    debug_transform = args.debug_sanity or args.debug_transform
    debug_index = args.debug_sanity or args.debug_index
    debug_pairs = args.debug_sanity or args.debug_pairs

    if args.run_self_tests or args.debug_sanity:
        run_self_tests(args.device, args.point_chunk_size, args.face_chunk_size)

    payload = load_scene_payload(args.scene_json)
    payload_by_id = scene_payload_by_id(payload)
    requested_scene_ids = args.scene_id or load_scene_ids_from_payload(payload)
    scene_dirs = find_scene_dirs(root, requested_scene_ids)[: args.max_scenes]

    if not scene_dirs:
        raise ValueError(f"No scenes found under {root}")

    collided_objects = 0
    total_objects = 0
    collided_scenes = 0
    per_scene = []

    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        flags, meshes = evaluate_scene(
            scene_dir,
            args.method,
            args.device,
            args.backend,
            args.point_chunk_size,
            args.face_chunk_size,
            args.transform_source,
            payload_by_id.get(scene_dir.name),
            args.size_scale,
            debug_transform,
            debug_index,
            debug_pairs,
        )
        scene_collided_objects = int(flags.sum())
        scene_total_objects = len(meshes)
        scene_has_collision = scene_collided_objects > 0

        collided_objects += scene_collided_objects
        total_objects += scene_total_objects
        collided_scenes += int(scene_has_collision)

        print(f"scene {scene_idx}/{len(scene_dirs)}: {scene_dir.name}")
        print(flags.astype(int).tolist())
        print(
            "  collided objects: "
            f"{scene_collided_objects}/{scene_total_objects}"
        )

        per_scene.append(
            {
                "scene_id": scene_dir.name,
                "object_count": scene_total_objects,
                "collided_object_count": scene_collided_objects,
                "has_collision": scene_has_collision,
                "objects": [
                    {
                        "path": str(path),
                        "collides": bool(flag),
                        "bounds_min": mesh.bounds[0].tolist(),
                        "bounds_max": mesh.bounds[1].tolist(),
                    }
                    for flag, (path, mesh) in zip(flags, meshes)
                ],
            }
        )

    col_obj = collided_objects / total_objects
    col_scene = collided_scenes / len(scene_dirs)

    print("overlap object: ", col_obj, "cnt ", collided_objects, "/", total_objects)
    print("overlap scene rate: ", col_scene)
    print("ColObj:", col_obj)
    print("ColScene:", col_scene)

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": args.method,
                    "backend": args.backend,
                    "transform_source": args.transform_source,
                    "size_scale": args.size_scale,
                    "object_mesh_root": str(root),
                    "scene_count": len(scene_dirs),
                    "object_count": total_objects,
                    "collided_object_count": collided_objects,
                    "collided_scene_count": collided_scenes,
                    "ColObj": col_obj,
                    "ColScene": col_scene,
                    "per_scene": per_scene,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
