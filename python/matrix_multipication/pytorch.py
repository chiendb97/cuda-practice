import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M',
                        type=int,
                        required=True,
                        help='M')
    parser.add_argument('--N',
                        type=int,
                        required=True,
                        help='N')
    parser.add_argument('--K',
                        type=int,
                        required=True,
                        help='K')

    return parser.parse_args()


def main():
    args = get_args()
    A = torch.rand(size=(args.M, args.K), device='cuda', dtype=torch.float16)
    B = torch.rand(size=(args.K, args.N), device='cuda', dtype=torch.float16)
    C = torch.zeros(size=(args.M, args.N), device='cuda', dtype=torch.float16)

    C = A * B

    torch.matmul(A, B, out=C)


if __name__ == '__main__':
    main()
