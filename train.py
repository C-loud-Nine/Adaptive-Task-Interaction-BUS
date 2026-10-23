"""Training script for multi-task breast ultrasound model."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import config
from model import adaptive_interaction_model
from modules import calibrate_aiw
from loss import enhanced_lesion_focus_loss, enhanced_multi_modal_focal_loss
from data_loader import prepare_datasets


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================
class CompositeMetric(Callback):
    """Composite validation score: 0.7 * IoU + 0.3 * accuracy."""
    def on_epoch_end(self, epoch, logs=None):
        seg_iou = logs.get("val_segmentation_output_mean_io_u", 0)
        clf_acc = logs.get("val_classification_output_accuracy", 0)
        val_combined = 0.7 * seg_iou + 0.3 * clf_acc
        logs["val_combined"] = val_combined
        print(f"\nEpoch {epoch+1}: Combined Metric = {val_combined:.4f}")


class CosineDecayScheduler(Callback):
    def __init__(
        self,
        initial_lr: float,
        total_epochs: int,
        alpha: float = 0.03,
        monitor: str = "val_combined",
        factor: float = 0.5,
        patience: int = 7,
        cooldown: int = 2,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.initial_lr    = initial_lr
        self.total_epochs  = total_epochs
        self.alpha         = alpha
        self.monitor       = monitor
        self.factor        = factor
        self.patience      = patience
        self.cooldown      = cooldown
        self.min_lr        = min_lr

        self.best               = -np.inf
        self.wait               = 0
        self.cooldown_counter   = 0
        self.lr_reduction_factor = 1.0   # accumulates all plateau multipliers

    def _cosine_lr(self, epoch):
        """Pure cosine value for this epoch, before plateau scaling."""
        progress     = epoch / float(self.total_epochs)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.initial_lr * (self.alpha + (1.0 - self.alpha) * cosine_decay)

    def on_epoch_begin(self, epoch, logs=None):
        # Cosine baseline × accumulated plateau reductions, clamped to min_lr
        new_lr = max(self._cosine_lr(epoch) * self.lr_reduction_factor, self.min_lr)
        self.model.optimizer.learning_rate.assign(new_lr)
        print(f"\nEpoch {epoch+1}: LR = {new_lr:.7f}  "
              f"(cosine × {self.lr_reduction_factor:.4f})")

    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        # Count down cooldown — at end of epoch, not beginning
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return  # don't touch wait counter during cooldown

        if current > self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Store the reduction — on_epoch_begin of next epoch applies it
                self.lr_reduction_factor = max(
                    self.lr_reduction_factor * self.factor,
                    self.min_lr / self.initial_lr   # don't let factor push below min_lr
                )
                current_lr = float(self.model.optimizer.learning_rate)
                projected  = max(self._cosine_lr(epoch + 1) * self.lr_reduction_factor,
                                 self.min_lr)
                print(f"\nEpoch {epoch+1}: {self.monitor} plateaued "
                      f"({self.wait} epochs). "
                      f"Next LR → {projected:.7f}  "
                      f"[reduction_factor now {self.lr_reduction_factor:.4f}]")
                self.cooldown_counter = self.cooldown
                self.wait = 0


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    import argparse, random, csv, datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="busi", choices=["busi", "busi_whu"])
    ap.add_argument("--aiw-mode", default=config.AIW_MODE,
                    choices=["adaptive", "off", "fixed", "constant"])
    ap.add_argument("--mscf", default="on", choices=["on", "off"])
    ap.add_argument("--tim", default="on", choices=["on", "off"])
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE,
                    help="training seed; the split seed stays config.RANDOM_STATE")
    args = ap.parse_args()
    config.AIW_MODE = args.aiw_mode
    config.USE_MSCF = args.mscf == "on"
    config.USE_TIM = args.tim == "on"
    variant = (f"mscf-{args.mscf}_tim-{args.tim}_aiw-{args.aiw_mode}"
               if (args.mscf, args.tim) != ("on", "on") or args.aiw_mode != "adaptive"
               else "full")
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    print("=" * 80)
    print(f"Multi-Task Breast Ultrasound Training | {args.dataset} | seed {args.seed} "
          f"| variant {variant} | input {config.INPUT_SCALE} | IoU {config.SELECTION_IOU}")
    print("=" * 80)
    
    # Load data
    path = config.BUSI_WHU_PATH if args.dataset == "busi_whu" else config.BUSI_PATH
    num_classes = len(config.WHU_CLASSES) if args.dataset == "busi_whu" else config.NUM_CLF_CLASSES
    data = prepare_datasets(path, dataset=args.dataset)
    
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_train'],
        {
            "segmentation_output": data['mask_train'],
            "classification_output": data['y_train']
        }
    )).shuffle(3000).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_val'],
        {
            "segmentation_output": data['mask_val'],
            "classification_output": data['y_val']
        }
    )).batch(config.VAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    print("\nBuilding model...")
    model = adaptive_interaction_model(
        input_size=config.INPUT_SIZE,
        num_seg_classes=config.NUM_SEG_CLASSES,
        num_clf_classes=num_classes,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LR,
            beta_1=config.ADAM_BETA_1,
            beta_2=config.ADAM_BETA_2,
            epsilon=config.ADAM_EPSILON,
            global_clipnorm=config.GLOBAL_CLIPNORM
        ),
        loss={
            "segmentation_output": enhanced_lesion_focus_loss,
            "classification_output": enhanced_multi_modal_focal_loss
        },
        metrics={
            "segmentation_output": [
                (tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5,
                                            name="mean_io_u")
                 if config.SELECTION_IOU == "binary_iou"
                 else tf.keras.metrics.MeanIoU(num_classes=2, name="mean_io_u")),
                tf.keras.metrics.BinaryAccuracy(name="bin_acc")
            ],
            "classification_output": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        },
        loss_weights={
            "segmentation_output": config.SEG_WEIGHT,
            "classification_output": config.CLF_WEIGHT
        }
    )
    
    # Callbacks
    composite_metric = CompositeMetric()
    lr_scheduler = CosineDecayScheduler(
        initial_lr=config.INITIAL_LR,
        total_epochs=config.EPOCHS,
        alpha=config.COSINE_ALPHA,
        monitor=config.MONITOR,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        cooldown=config.REDUCE_LR_COOLDOWN,
        min_lr=config.MIN_LR
    )
    early_stop = EarlyStopping(
        monitor=config.MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode=config.MODE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Warmup
    print("\n" + "=" * 80)
    print("WARMUP PHASE")
    print("=" * 80)
    model.fit(
        train_dataset.take(100),
        validation_data=val_dataset,
        epochs=config.WARMUP_EPOCHS,
        callbacks=[lr_scheduler],
        verbose=1
    )
    
    # Main training
    print("\n" + "=" * 80)
    print("MAIN TRAINING")
    print("=" * 80)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        initial_epoch=config.WARMUP_EPOCHS,
        callbacks=[
            composite_metric,
            early_stop,
            lr_scheduler,
        ],
        verbose=1
    )
    
    # Fix AIW's inference means: mean descriptor over the training set,
    # computed in inference mode with the final (restored) weights.
    aiw_stats = calibrate_aiw(model, data['X_train'], batch_size=config.VAL_BATCH_SIZE)
    for name, (ms, mc) in aiw_stats.items():
        print(f"AIW calibration {name}: m_seg = {ms:.6g}, m_clf = {mc:.6g}")

    # Save model: one file per dataset, AIW mode and seed.
    model_path = os.path.join(config.CHECKPOINT_DIR,
                              f'best_{args.dataset}_{variant}_s{args.seed}.keras')
    model.save(model_path)
    stopped_early = early_stop.stopped_epoch > 0
    print(f"\nModel saved to {model_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)
    test_dataset = tf.data.Dataset.from_tensor_slices((
        data['X_test'],
        {
            "segmentation_output": data['mask_test'],
            "classification_output": data['y_test']
        }
    )).batch(config.VAL_BATCH_SIZE)
    
    test_results = model.evaluate(test_dataset, verbose=1)
    print("\nTest Results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"  {name}: {value:.4f}")
    test_acc = dict(zip(model.metrics_names, test_results)).get(
        "classification_output_accuracy", float("nan"))
    n_correct = int(round(test_acc * len(data['X_test'])))
    import hashlib
    with open(model_path, "rb") as fh:
        ckpt_sha = hashlib.sha256(fh.read()).hexdigest()[:16]

    # Run ledger: one row per run.
    ledger = os.path.join(config.RESULTS_DIR, "run_ledger.csv")
    new = not os.path.exists(ledger)
    with open(ledger, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "dataset", "training_seed", "split_seed", "variant",
                        "input_scale", "selection_iou", "n_test", "n_correct",
                        "stopped_early", "checkpoint", "checkpoint_sha256_16"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), args.dataset,
                    args.seed, config.RANDOM_STATE, variant, config.INPUT_SCALE,
                    config.SELECTION_IOU, len(data['X_test']), n_correct,
                    int(stopped_early), model_path, ckpt_sha])
    print(f"Run recorded in {ledger}; run evaluate.py on {model_path} for per-image metrics.")


if __name__ == "__main__":
    main()
