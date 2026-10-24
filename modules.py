import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MultiScaleContextFusion(layers.Layer):
    """Multi-scale context fusion (MSCF): dilated separable convolutions
    (rates 1, 2, 4) fused by a per-image softmax over scales."""
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.dilation_convs = [
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=1, name=f'{self.name}_sep_conv_1'),
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=2, name=f'{self.name}_sep_conv_2'),
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=4, name=f'{self.name}_sep_conv_4'),
        ]
        self.scale_attention = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.channels // self.reduction_ratio, activation='relu'),
            layers.Dense(3, activation='softmax'),
        ], name=f'{self.name}_scale_attention')
        self.cross_scale_fusion = layers.Conv2D(self.channels, 1, padding='same', name=f'{self.name}_fusion')
        super().build(input_shape)

    def call(self, inputs):
        scale_features = [conv(inputs) for conv in self.dilation_convs]
        scale_weights = self.scale_attention(inputs)
        scale_weights = tf.reshape(scale_weights, [-1, 1, 1, 3])
        weighted = [scale_features[i] * scale_weights[:, :, :, i:i+1] for i in range(3)]
        return self.cross_scale_fusion(tf.add_n(weighted))

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'reduction_ratio': self.reduction_ratio})
        return config


class DualPathAttention(layers.Layer):
    """CBAM-style channel and spatial attention used in the encoder blocks.

    An optional task-context gate is applied when task_context is given; the
    paper's model calls this layer without it.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.channel_path = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(max(8, self.channels // 16), activation='relu'),
            layers.Dense(self.channels, activation='sigmoid'),
        ], name=f'{self.name}_channel_path')
        self.spatial_path = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name=f'{self.name}_spatial')
        self.cross_task_context = layers.Dense(self.channels // 8, activation='relu', name=f'{self.name}_cross_task')
        self.task_gate_dense = layers.Dense(self.channels, name=f'{self.name}_task_gate')
        super().build(input_shape)

    def call(self, inputs, task_context=None):
        channel_weights = tf.reshape(self.channel_path(inputs), [-1, 1, 1, self.channels])
        spatial_input = tf.concat([
            tf.reduce_max(inputs, axis=3, keepdims=True),
            tf.reduce_mean(inputs, axis=3, keepdims=True),
        ], axis=3)
        spatial_weights = self.spatial_path(spatial_input)
        if task_context is not None:
            task_adapt = tf.reshape(
                self.cross_task_context(task_context), [-1, 1, 1, self.channels // 8]
            )
            task_gate = tf.reshape(
                tf.sigmoid(self.task_gate_dense(task_adapt)), [-1, 1, 1, self.channels]
            )
            channel_weights = channel_weights * task_gate
        return inputs * channel_weights * spatial_weights

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config


class ResidualBlock(layers.Layer):
    """Residual block with multi-scale fusion (MSCF) and optional dual-path attention.

    use_mscf=False gives the plain residual block used by the ablation rows
    without MSCF.
    """
    def __init__(self, channels, use_attention=True, use_mscf=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.use_attention = use_attention
        self.use_mscf = use_mscf

    def build(self, input_shape):
        self.conv1 = layers.SeparableConv2D(self.channels, 3, padding='same', activation='relu', name=f'{self.name}_conv1')
        self.bn1   = layers.BatchNormalization(name=f'{self.name}_bn1')
        self.conv2 = layers.SeparableConv2D(self.channels, 3, padding='same', name=f'{self.name}_conv2')
        self.bn2   = layers.BatchNormalization(name=f'{self.name}_bn2')
        if self.use_mscf:
            self.multiscale = MultiScaleContextFusion(self.channels, name=f'{self.name}_multiscale')
        if self.use_attention:
            self.attention = DualPathAttention(self.channels, name=f'{self.name}_attention')
        self.residual_proj = layers.Conv2D(self.channels, 1, padding='same', name=f'{self.name}_proj')
        super().build(input_shape)

    def call(self, inputs, task_context=None):
        residual = inputs
        x = self.bn1(self.conv1(inputs))
        x = self.bn2(self.conv2(x))
        if self.use_mscf:
            x = self.multiscale(x)
        if self.use_attention:
            x = self.attention(x, task_context)
        if residual.shape[-1] != self.channels:
            residual = self.residual_proj(residual)
        return tf.keras.activations.relu(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'use_attention': self.use_attention,
                       'use_mscf': self.use_mscf})
        return config


class AttentionGate(layers.Layer):
    """Attention gate with multi-scale context for skip connections."""
    def __init__(self, inter_channels, **kwargs):
        super().__init__(**kwargs)
        self.inter_channels = inter_channels

    def build(self, input_shape):
        self.x_transform = layers.Conv2D(self.inter_channels, 1, padding='same', name=f'{self.name}_x_transform')
        self.g_transform = layers.Conv2D(self.inter_channels, 1, padding='same', name=f'{self.name}_g_transform')
        self.context_attention = tf.keras.Sequential([
            layers.SeparableConv2D(self.inter_channels, 3, padding='same', dilation_rate=1),
            layers.SeparableConv2D(self.inter_channels, 3, padding='same', dilation_rate=2),
            layers.Conv2D(1, 1, padding='same', activation='sigmoid'),
        ], name=f'{self.name}_context_attention')
        super().build(input_shape)

    def call(self, x, g):
        theta_x = self.x_transform(x)
        phi_g   = self.g_transform(g)
        if x.shape[1] != g.shape[1] or x.shape[2] != g.shape[2]:
            phi_g = tf.image.resize(phi_g, [x.shape[1], x.shape[2]])
        combined = tf.nn.relu(theta_x + phi_g)
        return x * self.context_attention(combined)

    def get_config(self):
        config = super().get_config()
        config.update({'inter_channels': self.inter_channels})
        return config


class TaskInteractionModule(layers.Layer):
    """Task Interaction Module (TIM), Eqs. (1)-(3) of the paper.

    Segmentation -> classification: gated residual update of the
    classification vector by a pooled decoder summary.
    Classification -> segmentation: channel modulation
    mu = 1 + 0.7 * g * p of the decoder features.
    """
    def __init__(self, seg_channels=192, clf_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.seg_channels = seg_channels
        self.clf_channels = clf_channels

    def build(self, input_shape):
        self.seg_to_clf = tf.keras.Sequential([
            layers.Conv2D(self.seg_channels // 2, 1, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.clf_channels, activation='relu'),
            layers.BatchNormalization(),
        ], name=f'{self.name}_seg_to_clf')
        self.clf_to_seg = tf.keras.Sequential([
            layers.Dense(self.seg_channels, activation='sigmoid'),
            layers.Reshape((1, 1, self.seg_channels)),
        ], name=f'{self.name}_clf_to_seg')
        self.seg_gate = layers.Dense(self.seg_channels, activation='sigmoid', name=f'{self.name}_seg_gate')
        self.clf_gate = layers.Dense(self.clf_channels, activation='sigmoid', name=f'{self.name}_clf_gate')
        self.seg_gap  = layers.GlobalAveragePooling2D()
        super().build(input_shape)

    def call(self, seg_features, clf_features, training=None):
        # Seg → Clf
        seg_context      = self.seg_to_clf(seg_features)
        clf_gate_weights = self.clf_gate(seg_context)
        enhanced_clf     = clf_features + clf_gate_weights * seg_context

        # Clf → Seg
        clf_context      = self.clf_to_seg(clf_features)
        seg_global       = tf.reshape(self.seg_gate(self.seg_gap(seg_features)), [-1, 1, 1, self.seg_channels])
        modulation       = 1.0 + 0.7 * seg_global * clf_context
        enhanced_seg     = seg_features * modulation

        return enhanced_seg, enhanced_clf

    def get_config(self):
        config = super().get_config()
        config.update({'seg_channels': self.seg_channels, 'clf_channels': self.clf_channels})
        return config


class AdaptiveInteractionWeighting(layers.Layer):
    """Adaptive Interaction Weighting (AIW).

    Blends pre-interaction and TIM-enhanced features with two coefficients
    produced by a softmax, so w_seg + w_clf = 1 at every decoder level
    (Eqs. (4)-(8) of the paper).

    Each activation-dispersion descriptor is divided by a mean m_t (Eq. 5).
    During training, and for validation during training, m_t is the mean over
    the current batch, exactly as in the original formulation. After training,
    calibrate_aiw() fixes m_t to the mean of the descriptor over the training
    set computed in inference mode; a calibrated layer then uses that fixed
    value at inference, so test predictions are independent of batch size,
    order and composition.
    """
    def __init__(self, mode=None, **kwargs):
        kwargs.pop("momentum", None)          # accepted for older configs
        super().__init__(**kwargs)
        import config as _cfg
        self.mode = mode or getattr(_cfg, "AIW_MODE", "adaptive")
        if self.mode not in ("adaptive", "off", "fixed", "constant"):
            raise ValueError(f"unknown AIW mode: {self.mode}")

    def build(self, input_shape):
        # The weight network exists only in "adaptive" mode; unused, unbuilt
        # sub-layers would make model.trainable_variables raise.
        if self.mode == "adaptive":
            self.weight_network = tf.keras.Sequential([
                layers.Dense(32, activation='relu'),
                layers.Dense(2,  activation='softmax'),
            ], name=f'{self.name}_weight_network')
            # Fixed inference means, set by calibrate_aiw (not trained by SGD).
            self.pop_seg = self.add_weight(name="pop_seg", shape=(),
                                           initializer="zeros", trainable=False)
            self.pop_clf = self.add_weight(name="pop_clf", shape=(),
                                           initializer="zeros", trainable=False)
            self.calibrated = self.add_weight(name="calibrated", shape=(),
                                              initializer="zeros", trainable=False)
        if self.mode == "constant":
            self.const_logits = self.add_weight(
                name="const_logits", shape=(2,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def _activation_dispersion(self, features):
        """Activation dispersion of a feature tensor, shape [B, 1]: spatial
        variance averaged over channels for maps, variance for vectors."""
        if len(features.shape) == 4:
            mean      = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
            variance  = tf.reduce_mean(tf.square(features - mean), axis=[1, 2])
            return tf.reduce_mean(variance, axis=-1, keepdims=True)
        else:
            mean     = tf.reduce_mean(features, axis=1, keepdims=True)
            variance = tf.reduce_mean(tf.square(features - mean), axis=1, keepdims=True)
            return variance

    def set_population_stats(self, seg_mean, clf_mean):
        """Fix the inference means (called by calibrate_aiw after training)."""
        self.pop_seg.assign(float(seg_mean))
        self.pop_clf.assign(float(clf_mean))
        self.calibrated.assign(1.0)

    def compute_weights(self, seg_enhanced, clf_enhanced, training=False):
        """Return the two interaction weights, shape [B, 2], summing to 1.

        Also used by evaluate.py to export the per-image weights.
        """
        batch = tf.shape(seg_enhanced)[0]
        if self.mode == "off":
            return tf.ones([batch, 2])
        if self.mode == "fixed":
            return 0.5 * tf.ones([batch, 2])
        if self.mode == "constant":
            # softmax keeps the fixed-sum constraint of the adaptive gate
            return tf.tile(tf.nn.softmax(self.const_logits)[None, :], [batch, 1])

        seg_disp = self._activation_dispersion(seg_enhanced)
        clf_disp = self._activation_dispersion(clf_enhanced)

        # Eq. 5: batch mean while training (and before calibration); the fixed
        # calibrated mean at inference once calibrate_aiw has been run.
        if training is None:
            training = tf.keras.backend.learning_phase()
        if isinstance(training, (bool, int)):
            use_fixed = tf.logical_and(tf.constant(not bool(training)), self.calibrated > 0.0)
        else:
            use_fixed = tf.logical_and(tf.logical_not(tf.cast(training, tf.bool)),
                                       self.calibrated > 0.0)
        seg_mean = tf.where(use_fixed, self.pop_seg, tf.reduce_mean(seg_disp))
        clf_mean = tf.where(use_fixed, self.pop_clf, tf.reduce_mean(clf_disp))

        seg_disp_norm = seg_disp / (seg_mean + 1e-8)
        clf_disp_norm = clf_disp / (clf_mean + 1e-8)

        dispersions = tf.concat([seg_disp_norm, clf_disp_norm], axis=-1)
        return self.weight_network(dispersions)               # softmax: sums to 1

    def call(self, seg_base, seg_enhanced, clf_base, clf_enhanced, training=None):
        adaptive_weights = self.compute_weights(seg_enhanced, clf_enhanced, training=training)

        seg_weight = tf.reshape(adaptive_weights[:, 0:1], [-1, 1, 1, 1])
        clf_weight = tf.reshape(adaptive_weights[:, 1:2], [-1, 1])

        seg_final = seg_base + seg_weight * (seg_enhanced - seg_base)
        clf_final = clf_base + clf_weight * (clf_enhanced - clf_base)

        return seg_final, clf_final

    def get_config(self):
        config = super().get_config()
        config.update({'mode': self.mode})
        return config

def aiw_call_args(layer):
    """Return the four tensors an AIW layer was called with, in call order.

    Keras stores a multi-input call as node.call_args (positional) and
    node.call_kwargs (keyword); after loading a saved model the trailing
    arguments may appear as keywords.
    """
    node = layer._inbound_nodes[0]
    args = list(node.call_args)
    kwargs = node.call_kwargs or {}
    for key in ("seg_base", "seg_enhanced", "clf_base", "clf_enhanced"):
        if key in kwargs:
            args.append(kwargs[key])
    return args


def calibrate_aiw(model, images, batch_size=24):
    """Fix each adaptive AIW layer's inference means to the mean descriptor
    over `images` (the training set), computed in inference mode.

    Returns {layer name: (seg_mean, clf_mean)}; empty if the model has no
    adaptive AIW layer.
    """
    aiw_layers = [l for l in model.layers
                  if isinstance(l, AdaptiveInteractionWeighting) and l.mode == "adaptive"]
    if not aiw_layers:
        return {}
    stats = {}
    # Level by level (model.layers is in topological order, d1 to d4), so each
    # level is measured with all upstream levels already calibrated.
    for layer in aiw_layers:
        ins = aiw_call_args(layer)
        probe = tf.keras.Model(model.inputs, [layer._activation_dispersion(ins[1]),
                                              layer._activation_dispersion(ins[3])])
        v_seg, v_clf = probe.predict(images, batch_size=batch_size, verbose=0)
        seg_mean, clf_mean = float(np.mean(v_seg)), float(np.mean(v_clf))
        layer.set_population_stats(seg_mean, clf_mean)
        stats[layer.name] = (seg_mean, clf_mean)
    return stats
