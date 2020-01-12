from __future__ import division, print_function

import argparse
import os
import time

import numpy as np
import scipy

from utils import get_files, get_img, save_img, resize_to, center_crop
from utils import preserve_colors_np
from wct import WCT

# python3 stylize.py --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 --style-size 512 --alpha 0.8 --style-path ./styles/bubbly_0060.jpg --content-path gilbert.jpg --out-path result


parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str,
                    help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-paths', nargs='+', type=str, dest='style_paths', help='Style images')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512",
                    default=0)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('-r', '--random', type=int, help="Choose # of random subset of images from style folder", default=0)
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--beta', type=float, help="Beta blend value", default=0.5)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--adain', action='store_true', help="Use AdaIN instead of WCT", default=False)

## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

args = parser.parse_args()


def stylize_output(wct_model, content_img, styles):
    if args.crop_size > 0:
        styles = [
            center_crop(style) for style in styles
        ]
    if args.keep_colors:
        styles = [
            preserve_colors_np(style, content_img) for style in styles
        ]

    # Run the frame through the style network
    stylized = content_img
    for _ in range(args.passes):
        stylized = wct_model.predict(stylized, styles)

    # Stitch the style + stylized output together, but only if there's one style image
    if args.concat:
        # Resize style img to same height as frame
        style_img_resized = scipy.misc.imresize(styles[0], (stylized.shape[0], stylized.shape[0]))
        stylized = np.hstack([style_img_resized, stylized])

    return stylized


def main():
    start = time.time()
    if len(args.style_paths) > 2:
        raise Exception('Maximum number of styles should be 2')

    # Load the WCT model
    wct_model = WCT(checkpoints=args.checkpoints,
                    relu_targets=args.relu_targets,
                    vgg_path=args.vgg_path,
                    device=args.device,
                    ss_patch_size=args.ss_patch_size,
                    ss_stride=args.ss_stride,
                    alpha=args.alpha,
                    beta=args.beta)

    os.makedirs(args.out_path, exist_ok=True)

    content_img = get_img(args.content_path, args.content_size)
    styles = [
        get_img(path, args.style_size)
        for path in args.style_paths
    ]

    _, content_ext = os.path.splitext(args.content_path)
    output_filename = os.path.join(args.out_path, "result.jpg")#f'{args.content_path}_{args.style_path_a}.{content_ext}')
    output = stylize_output(wct_model, content_img, styles)
    save_img(output_filename, output)

    print("Finished stylizing in {}s".format(time.time() - start))


if __name__ == '__main__':
    main()
