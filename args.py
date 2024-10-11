import argparse

def parse_args(argv):
    parser = setup_parser()
    args = parser.parse_args(argv)
    return args

def setup_parser():
    parser = argparse.ArgumentParser(description='NeuSDFusion')

    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_epochs",
        "-e",
        type=int,
        default=30,
        help="Stage 1: Max Epochs per Object",
    )
    parser.add_argument("--num_on_surface_points", type=int, default=1000, help="Stage 1: Number of on-surface points to sample from mesh")
    parser.add_argument("--num_off_surface_points", type=int, default=1000, help="Stage 1: Number of off-surface points to sample from mesh")
    parser.add_argument(
        "--train_batch_size", type=int, default=2000, help="Stage 1: Num of sampled points per forward pass. This should be the same as total number of points sampled on the object (for now lol)."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Stage 1: Learning rate for training NeuSDF triplanes"
    )
    # TODO: Not implement yet, hard-coded
    parser.add_argument(
        "--sdf_loss_weight", type=float, default=1.0, help="Stage 1: Weight for the SDF loss"
    )
    # TODO: Not implement yet, hard-coded
    parser.add_argument(
        "--normal_loss_weight", type=float, default=0.1, help="Stage 1: Weight for the normal loss"
    )
    # TODO: Not implement yet, hard-coded
    parser.add_argument(
        "--eikonal_loss_weight", type=float, default=0.1, help="Stage 1: Weight for the eikonal loss"
    )
    parser.add_argument(
        "--resolution", type=int, default=128, help="Stage 1 & 2: Resolution of the triplane"
    )
    # TODO: Not implement yet, hard-coded
    parser.add_argument("--mlp_hidden_layers", type=int, default=2, help="Stage 1 & 2: Number of hidden layers in triplane mlp")
    parser.add_argument(
        "--mlp_hidden_dim", type=int, default=256, help="Stage 1 & 2: Hidden dimension for the MLPDecoder"
    )
    parser.add_argument("--mlp_output_dim", type=int, default=1, help="Stage 1 & 2: Output dimension for the MLPDecoder")

    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--shapenet_dir", type=str, default='/home/harsha/Documents/watertight_shapenet/02691156', help="Stage 1: Input directory for training data"
    )
    parser.add_argument(
        "--triplane_dir", type=str, default='/home/harsha/Documents/triplanes/02691156', help="Stage 1: Output directory for model and logs. Use as input directory for Stage 2."
    )

    parser.add_argument("--stage2_train_batch_size", type=int, default=2, help="Stage 2: Train batch size for autoencoder (number of objects)")
    parser.add_argument("--stage2_max_epochs", type=int, default=10, help="Stage 2: Max epochs for autoencoder")
    parser.add_argument("--stage2_learning_rate", type=float, default=1e-5, help="Stage 2: Learning rate for autoencoder")

    # Transformer Encoder HyperParams
    parser.add_argument("--gc_kernel_size", type=int, default=3, help="Stage 2: Kernel size for grouped down convolution")
    parser.add_argument("--gc_stride", type=int, default=2, help="Stage 2: Stride for grouped down convolution")
    parser.add_argument("--embedding_dim", type=int, default=96, help="Stage 2: Dimension of the Transformer Encoder embedding vector")
    parser.add_argument("--d_model", type=int, default=96, help="Stage 2: Dimension of the Transformer Encoder input vector")
    parser.add_argument("--n_heads", type=int, default=2, help="Stage 2: Number of heads in the transformer")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Stage 2: Number of transformer encoder layers")

    # Transformer Decoder HyperParams
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Stage 2: Number of transformer decoder layers")

    return parser

