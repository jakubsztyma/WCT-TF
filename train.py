from __future__ import print_function, division

import argparse
import time
import os

import numpy as np
import tensorflow as tf

from model import WCTModel
from utils import get_files, get_img_random_crop

# python3 train.py --relu-target relu2_1 --content ./content --checkpoint ./models --max-iter=10


mse = tf.compat.v1.losses.mean_squared_error

kwargs_list = [
    # Directories
    dict(type=str, dest='checkpoint', help='Checkpoint save dir', required=True),
    dict(type=str, dest='log_path', help='Logging dir path'),
    dict(nargs='+', type=str, dest="relu_target", help='Target VGG19 relu layer to decode from, e.g. relu4_1', required=True, ),
    dict(type=str, dest='content_path', help='Content images folder', required=True),
    dict(type=str, dest='val_path', help='Validation images folder', default=None),
    dict(type=str, dest='vgg_path', help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7'),

    # Loss weights
    dict(type=float, dest='feature_weight', help='Feature loss weight', default=1),
    dict(type=float, dest='pixel_weight', help='Pixel reconstruction loss weight', default=1),
    dict(type=float, dest='tv_weight', help='Total variation loss weight', default=0),

    # Train opts
    dict(type=float, dest='learning_rate', help='Learning rate', default=1e-4),
    dict(type=float, dest='lr_decay', help='Learning rate decay', default=0),
    dict(type=int, dest='max_iter', help='Max # of training iterations', default=16000),
    dict(type=int, dest='batch_size', help='Batch size', default=8),
    dict(type=int, dest='save_iter', help='Checkpoint save frequency', default=200),
    dict(type=int, dest='summary_iter', help='Summary write frequency', default=20),
    dict(type=int, dest='max_to_keep', help='Max # of checkpoints to keep around', default=10),
]

parser = argparse.ArgumentParser()
for kwargs in kwargs_list:
    kwarg_name = kwargs['dest'].replace('_', '-')
    parser.add_argument(f"--{kwarg_name}", **kwargs)
args = parser.parse_args()


def batch_gen(folder, batch_shape):
    '''Resize images to 512, randomly crop a 256 square, and normalize'''
    files = np.asarray(get_files(folder))
    while True:
        X_batch = np.zeros(batch_shape, dtype=np.float32)

        idx = 0

        while idx < batch_shape[0]:  # Build batch sample by sample
            try:
                f = np.random.choice(files)

                X_batch[idx] = get_img_random_crop(f, resize=512, crop=256).astype(np.float32)
                X_batch[idx] /= 255.  # Normalize between [0,1]

                assert (not np.isnan(X_batch[idx].min()))
            except Exception as e:
                # Do not increment idx if we failed 
                print(e)
                continue
            idx += 1

        yield X_batch


def train():
    batch_shape = (args.batch_size, 256, 256, 3)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    model = WCTModel(relu_targets=args.relu_target, vgg_path=args.vgg_path)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    content_images = batch_gen(args.content_path, batch_shape)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    test_loss = tf.keras.metrics.Mean()
    train_loss = tf.keras.metrics.Mean()
    checkpoint = tf.train.Checkpoint(model=model.decoders[0])
    manager = tf.train.CheckpointManager(checkpoint, args.checkpoint, max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions, decoded_encoded, content_encoded = model(images, training=True)
            pixel_loss = loss_object(labels, predictions)
            feature_loss = loss_object(content_encoded, decoded_encoded)
            loss = pixel_loss + feature_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def test_step(images, labels):
        predictions, decoded_encoded, content_encoded = model(images, training=True)
        t_pixel_loss = loss_object(labels, predictions)
        t_feature_loss = loss_object(content_encoded, decoded_encoded)
        t_loss = t_pixel_loss + t_feature_loss

        test_loss(t_loss)

    for iteration in range(args.max_iter):
        start = time.time()

        batch = next(content_images)
        train_step(batch, batch)

        test_batch = next(content_images)
        elapsed = time.time() - start
        print(f"Iteration {iteration} took {elapsed:.4}s")

        if iteration % 5 == 0:
            test_step(test_batch, test_batch)

            print(f"After iteration {iteration}:")
            print(f"Test loss:  {float(test_loss.result())}")
            print(f"Train loss:  {float(train_loss.result())}")

            # Reset the metrics for the next epoch
            train_loss.reset_states()
            test_loss.reset_states()

    # Last save
    manager.save()
    print(f"Model saved in file: {manager.latest_checkpoint}")


if __name__ == '__main__':
    train()
