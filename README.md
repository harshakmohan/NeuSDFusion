# NeuSDF
Implementation of NeuSDF Paper.

Dataset downloaded from: https://huggingface.co/datasets/ShapeNet/ShapeNetCore

Repo used to convert Meshes into Watertight Manifolds: https://github.com/hjwdzh/Manifold/tree/master

To convert the Shapenet dataset to a Watertight Manifold, I cloned the Manifold repo, built it locally, then wrote a python script to convert the ShapeNet data that was on my local harddrive into Watertight obj files that are then stored in a separate directory.

Triplane representation is described here: https://arxiv.org/pdf/2112.07945

