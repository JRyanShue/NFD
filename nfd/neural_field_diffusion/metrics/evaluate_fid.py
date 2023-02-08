import argparse
import os

from pytorch_fid.fid_score import calculate_fid_given_paths


def main(args):

    fid = 0
    for view_index in range(20):
        gt_view_dir = os.path.join(args.gt_dir, f"view_{view_index}")
        gen_view_dir = os.path.join(args.gen_dir, f"view_{view_index}")
        view_score = calculate_fid_given_paths([gt_view_dir, gen_view_dir],
                                               args.batch_size,
                                               args.device,
                                               dims=2048,
                                               num_workers=8)
        fid += view_score
    fid /= 20
    print(f"FID: {fid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(args)
