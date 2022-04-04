import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='number of hidden dimensions')
    
    parser.add_argument('--val_size', type=int, default=5000,
                        help='size of validation set')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loader')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()