"""
complexity.py - parameter counts and FLOPs at 224 x 224 for one image.

Usage:
    python complexity.py                       # builds the model from config
    python complexity.py --checkpoint path.h5  # or loads a checkpoint
FLOPs are counted by the TensorFlow profiler on the frozen graph (one
multiply-add = 2 FLOPs).
"""
import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import config
from model import adaptive_interaction_model


def flops(model):
    spec = tf.TensorSpec([1, *config.INPUT_SIZE], tf.float32)
    fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(fn)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    info = tf.compat.v1.profiler.profile(graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(),
                                         cmd="op", options=opts)
    return info.total_float_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    if args.checkpoint:
        from evaluate import CUSTOM_OBJECTS
        model = tf.keras.models.load_model(args.checkpoint, compile=False,
                                           custom_objects=CUSTOM_OBJECTS)
    else:
        model = adaptive_interaction_model(input_size=config.INPUT_SIZE)
    total = model.count_params()
    trainable = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"parameters: total {total:,} | trainable {trainable:,}")
    print(f"FLOPs per 224x224 image: {flops(model) / 1e9:.2f} G")


if __name__ == "__main__":
    main()
