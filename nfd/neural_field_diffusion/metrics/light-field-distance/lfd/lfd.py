import argparse
import sys
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

SIMILARITY_TAG = b"SIMILARITY:"
CURRENT_DIR = Path(__file__).parent

GENERATED_FILES_NAMES = [
    "all_q4_v1.8.art",
    "all_q8_v1.8.art",
    "all_q8_v1.8.cir",
    "all_q8_v1.8.ecc",
    "all_q8_v1.8.fd",
]

OUTPUT_NAME_TEMPLATES = [
    "{}_q4_v1.8.art",
    "{}_q8_v1.8.art",
    "{}_q8_v1.8.cir",
    "{}_q8_v1.8.ecc",
    "{}_q8_v1.8.fd",
]


def find_similarity_in_logs(logs: bytes) -> float:
    """Get line from the logs where similarity is mentioned.

    Args:
        logs: Unprocessed logs from the docker container after a command was
            run.

    Returns:
        Similarity measure from the log.
    """
    logs = logs.split()
    similarity_line: Optional[bytes] = None
    for index, line in enumerate(logs):
        if line.startswith(SIMILARITY_TAG):
            similarity_line = logs[index + 1]
            break
    return float(similarity_line)


class MeshEncoder:
    """Class holding an object and preprocessing it using an external cmd."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        """Instantiate the class.

        It instantiates an empty, temporary folder that will hold any
        intermediate data necessary to calculate Light Field Distance.

        Args:
            vertices: np.ndarray of vertices consisting of 3 coordinates each.
            triangles: np.ndarray where each entry is a vector with 3 elements.
                Each element correspond to vertices that create a triangle.
        """
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        self.temp_dir_path = Path(tempfile.mkdtemp())
        self.file_name = uuid.uuid4()
        self.temp_path = self.temp_dir_path / "{}.obj".format(self.file_name)

        self.mesh.export(self.temp_path.as_posix())

    def get_path(self) -> str:
        """Get path of the object.

        Commands require that an object is represented without any extension.

        Returns:
            Path to the temporary object created in the file system that
            holds the Wavefront OBJ data of the object.
        """
        return self.temp_path.with_suffix("").as_posix()

    def align_mesh(self):
        """Create data of a 3D mesh to calculate Light Field Distance.

        It runs an external command that create intermediate files and moves
        these files to created temporary folder.

        Returns:
            None
        """
        process = subprocess.Popen(
            ["./3DAlignment", self.temp_path.with_suffix("").as_posix()],
            cwd=(CURRENT_DIR / "Executable").as_posix(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output, err = process.communicate()
        if len(err) > 0:
            print(err)
            sys.exit(1)

        for file, out_file in zip(
            GENERATED_FILES_NAMES, OUTPUT_NAME_TEMPLATES
        ):
            shutil.move(
                os.path.join((CURRENT_DIR / "Executable").as_posix(), file),
                (
                    self.temp_dir_path / out_file.format(self.file_name)
                ).as_posix(),
            )

    def __del__(self):
        shutil.rmtree(self.temp_dir_path.as_posix())


class LightFieldDistance:
    """Class that allows to calculate light field distance.

    It supports representing objects in the Wavefront OBJ format.
    """

    def __init__(self, verbose: bool = 0):
        """Instantiate the class.

        Args:
            verbose: Whether to display processing information performed step
                by step.
        """
        self.verbose = verbose

    def get_distance(
        self,
        vertices_1: np.ndarray,
        triangles_1: np.ndarray,
        vertices_2: np.ndarray,
        triangles_2: np.ndarray,
    ) -> float:
        """Calculate LFD between two meshes.

        These objects are taken as meshes from the Wavefront OBJ format. Hence
        vertices represent coordinates as a matrix Nx3, while `triangles`
        connects these vertices. Each entry in the `triangles` is a 3 element
        vector consisting of indices to appropriate vertices.

        Args:
            vertices_1: np.ndarray of vertices of the first object.
            triangles_1: np.ndarray of indices to vertices corresponding
                to particular indices connecting and forming a triangle.
            vertices_2: np.ndarray of vertices of the second object.
            triangles_2: np.ndarray of indices to vertices corresponding
                to particular indices connecting and forming a triangle. This
                parameter is for the second object.

        Returns:
            Light Field Distance between `object_1` and `object_2`.
        """
        mesh_1 = MeshEncoder(vertices_1, triangles_1)
        mesh_2 = MeshEncoder(vertices_2, triangles_2)

        if self.verbose:
            print("Aligning mesh 1 at {} ...".format(mesh_1.get_path()))
        mesh_1.align_mesh()

        if self.verbose:
            print("Aligning mesh 2 at {} ...".format(mesh_2.get_path()))
        mesh_2.align_mesh()

        if self.verbose:
            print("Calculating distances ...")

        process = subprocess.Popen(
            ["./Distance", mesh_1.get_path(), mesh_2.get_path()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=(CURRENT_DIR / "Executable").as_posix(),
        )

        output, err = process.communicate()
        lfd = find_similarity_in_logs(output)

        return lfd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Script that generates score for two shapes saved in Wavefront "
            "OBJ format"
        )
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to the first *.obj file in Wavefront OBJ format",
    )

    parser.add_argument(
        "file2",
        type=str,
        help="Path to the second *obj file in Wavefront OBJ format",
    )

    args = parser.parse_args()

    lfd_calc = LightFieldDistance(verbose=True)

    mesh_1: trimesh.Trimesh = trimesh.load(args.file1)
    mesh_2: trimesh.Trimesh = trimesh.load(args.file2)

    lfd = lfd_calc.get_distance(
        mesh_1.vertices, mesh_1.faces, mesh_2.vertices, mesh_2.faces
    )
    print("LFD: {:.4f}".format(lfd))


if __name__ == "__main__":
    main()
