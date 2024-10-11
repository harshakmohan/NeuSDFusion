# Utility to determine if an obj file contains a watertight manifold

import os
from collections import defaultdict

def read_obj_file(file_path):
    """Read the contents of an OBJ file and extract vertices and faces."""
    with open(file_path, 'r') as file:
        obj_data = file.readlines()

    vertices = []
    faces = []

    for line in obj_data:
        if line.startswith('v '):
            vertices.append(line.strip())
        elif line.startswith('f '):
            faces.append(line.strip())

    return vertices, faces

def create_edges(face):
    """Create edges from face vertices."""
    vertices = face.split()[1:]
    edges = []
    for i in range(len(vertices)):
        edge = tuple(sorted([vertices[i], vertices[(i + 1) % len(vertices)]]))
        edges.append(edge)
    return edges

def analyze_obj_file(file_path):
    """Analyze an OBJ file to check if it is a watertight manifold and provide basic metrics."""
    vertices, faces = read_obj_file(file_path)
    
    num_vertices = len(vertices)
    num_faces = len(faces)

    edge_count = defaultdict(int)

    # Count edges in the faces
    for face in faces:
        edges = create_edges(face)
        for edge in edges:
            edge_count[edge] += 1

    # Check for edges that are not shared by exactly two faces
    non_manifold_edges = {edge: count for edge, count in edge_count.items() if count != 2}
    num_non_manifold_edges = len(non_manifold_edges)

    is_watertight = (num_non_manifold_edges == 0)

    # Display results
    print(f"File: {file_path}")
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of faces: {num_faces}")
    print(f"Number of non-manifold edges: {num_non_manifold_edges}")
    print(f"Watertight: {'Yes' if is_watertight else 'No'}")

    if not is_watertight:
        print(f"Non-manifold edges: {non_manifold_edges}")

if __name__ == '__main__':
    # Example usage: replace 'path_to_your_obj_file.obj' with your actual OBJ file path
    file_path = 'path_to_your_obj_file.obj'
    analyze_obj_file(file_path)

