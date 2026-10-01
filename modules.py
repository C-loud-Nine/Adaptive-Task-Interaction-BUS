"""
Network modules for adaptive bidirectional decoder-level task interaction.

    MultiScaleContextFusion      (MSCF)  three dilated separable convolutions
                                         with softmax scale competition
    DualPathAttention                    channel and spatial attention
    ResidualBlock                        residual block wrapping MSCF
    AttentionGate                        gated skip connections
    TaskInteractionModule        (TIM)   bidirectional cross-task exchange
    AdaptiveInteractionWeighting (AIW)   per-image, per-level blend coefficient
"""

import tensorflow as tf
from tensorflow.keras import layers


class MultiScaleContextFusion(layers.Layer):
    """Three parallel dilated separable convolutions with softmax scale weighting."""

    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.dilation_convs = [
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=1,
                                   name=f'{self.name}_sep_conv_1'),
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=2,
                                   name=f'{self.name}_sep_conv_2'),
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=4,
                                   name=f'{self.name}_sep_conv_4'),
        ]
        self.scale_attention = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.channels // self.reduction_ratio, activation='relu'),
            layers.Dense(3, activation='softmax'),
        ], name=f'{self.name}_scale_attention')
        self.cross_scale_fusion = layers.Conv2D(self.channels, 1, padding='same',
                                                name=f'{self.name}_fusion')
        super().build(input_shape)

    def call(self, inputs):
        scale_features = [conv(inputs) for conv in self.dilation_convs]
        scale_weights = self.scale_attention(inputs)
        scale_weights = tf.reshape(scale_weights, [-1, 1, 1, 3])
        weighted = [scale_features[i] * scale_weights[:, :, :, i:i + 1] for i in range(3)]
        return self.cross_scale_fusion(tf.add_n(weighted))

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'reduction_ratio': self.reduction_ratio})
        return config


class DualPathAttention(layers.Layer):
    """Channel and spatial attention, in the manner of CBAM."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.channel_path = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(max(8, self.channels // 16), activation='relu'),
            layers.Dense(self.channels, activation='sigmoid'),
        ], name=f'{self.name}_channel_path')
        self.spatial_path = layers.Conv2D(1, 7, padding='same', activation='sigmoid',
                                          name=f'{self.name}_spatial')
        super().build(input_shape)

    def call(self, inputs):
        channel_weights = tf.reshape(self.channel_path(inputs), [-1, 1, 1, self.channels])
        spatial_input = tf.concat([
            tf.reduce_max(inputs, axis=3, keepdims=True),
            tf.reduce_mean(inputs, axis=3, keepdims=True),
        ], axis=3)
        spatial_weights = self.spatial_path(spatial_input)
        return inputs * channel_weights * spatial_weights

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config


class ResidualBlock(layers.Layer):
    """Residual block with multi-scale context fusion and optional attention."""

    def __init__(self, channels, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.use_attention = use_attention

    def build(self, input_shape):
        self.conv1 = layers.SeparableConv2D(self.channels, 3, padding='same',
                                            activation='relu', name=f'{self.name}_conv1')
        self.bn1 = layers.BatchNormalization(name=f'{self.name}_bn1')
        self.conv2 = layers.SeparableConv2D(self.channels, 3, padding='same',
                                            name=f'{self.name}_conv2')
        self.bn2 = layers.BatchNormalization(name=f'{self.name}_bn2')
        self.multiscale = MultiScaleContextFusion(self.channels, name=f'{self.name}_multiscale')
        if self.use_attention:
            self.attention = DualPathAttention(self.channels, name=f'{self.name}_attention')
        self.residual_proj = layers.Conv2D(self.channels, 1, padding='same',
                                           name=f'{self.name}_proj')
        super().build(input_shape)

    def call(self, inputs):
        residual = inputs
        x = self.bn1(self.conv1(inputs))
        x = self.bn2(self.conv2(x))
        x = self.multiscale(x)
        if self.use_attention:
            x = self.attention(x)
        if residual.shape[-1] != self.channels:
            residual = self.residual_proj(residual)
        return tf.keras.activations.relu(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'use_attention': self.use_attention})
        return config


class AttentionGate(layers.Layer):
    """Attention gate with multi-scale context for skip connections."""

    def __init__(self, inter_channels, **kwargs):
        super().__init__(**kwargs)
        self.inter_channels = inter_channels

    def build(self, input_shape):
        self.x_transform = layers.Conv2D(self.inter_channels, 1, padding='same',
                                         name=f'{self.name}_x_transform')
        self.g_transform = layers.Conv2D(self.inter_channels, 1, padding='same',
                                         name=f'{self.name}_g_transform')
        self.context_attention = tf.keras.Sequential([
            layers.SeparableConv2D(self.inter_channels, 3, padding='same', dilation_rate=1),
            layers.SeparableConv2D(self.inter_channels, 3, padding='same', dilation_rate=2),
            layers.Conv2D(1, 1, padding='same', activation='sigmoid'),
        ], name=f'{self.name}_context_attention')
        super().build(input_shape)

    def call(self, x, g):
        theta_x = self.x_transform(x)
        phi_g = self.g_transform(g)
        if x.shape[1] != g.shape[1] or x.shape[2] != g.shape[2]:
            phi_g = tf.image.resize(phi_g, [x.shape[1], x.shape[2]])
        combined = tf.nn.relu(theta_x + phi_g)
        return x * self.context_attention(combined)

    def get_config(self):
        config = super().get_config()
        config.update({'inter_channels': self.inter_channels})
        return config


class TaskInteractionModule(layers.Layer):
    """Bidirectional cross-task interaction (TIM).

    Seg -> Clf: 1x1 channel reduction, global average pooling, projection, then a
    gated residual into the classification vector.

    Clf -> Seg: the classification vector is projected to decoder width and scaled
    by a gate from the decoder's own global statistics. The resulting factor is

        mu = 1 + 0.7 * sigmoid(.) * sigmoid(.)   so   mu in [1, 1.7]

    which means the modulation is one-sided: a channel is left unchanged or
    amplified, never attenuated. Reversion toward the original features is
    achieved by AIW below, not by mu.
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
        self.seg_gate = layers.Dense(self.seg_channels, activation='sigmoid',
                                     name=f'{self.name}_seg_gate')
        self.clf_gate = layers.Dense(self.clf_channels, activation='sigmoid',
                                     name=f'{self.name}_clf_gate')
        self.seg_gap = layers.GlobalAveragePooling2D()
        super().build(input_shape)

    def call(self, seg_features, clf_features, training=None):
        # Seg -> Clf
        seg_context = self.seg_to_clf(seg_features)
        clf_gate_weights = self.clf_gate(seg_context)
        enhanced_clf = clf_features + clf_gate_weights * seg_context

        # Clf -> Seg
        clf_context = self.clf_to_seg(clf_features)
        seg_global = tf.reshape(self.seg_gate(self.seg_gap(seg_features)),
                                [-1, 1, 1, self.seg_channels])
        modulation = 1.0 + 0.7 * seg_global * clf_context   # mu in [1, 1.7]
        enhanced_seg = seg_features * modulation

        return enhanced_seg, enhanced_clf

    def get_config(self):
        config = super().get_config()
        config.update({'seg_channels': self.seg_channels, 'clf_channels': self.clf_channels})
        return config


class AdaptiveInteractionWeighting(layers.Layer):
    """Adaptive Interaction Weighting (AIW).

    Blends pre-interaction and post-interaction features with a coefficient
    computed per image and per decoder level. The two coefficients are produced
    by a softmax, so they sum to one: the branches share a single interaction
    budget at each level.

    The inputs are activation-dispersion statistics of the interacted features.
    These are descriptive statistics of the activations, not estimates of
    predictive uncertainty, and the module makes no uncertainty claim.

    Each dispersion value is divided by its batch mean, so a sample's coefficient
    depends on the other samples in its batch. This is a known limitation and is
    stated as such in the paper.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weight_network = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax'),
        ], name=f'{self.name}_weight_network')
        super().build(input_shape)

    def activation_dispersion(self, features):
        """Spatial variance averaged over channels (4-D), or plain variance (2-D)."""
        if len(features.shape) == 4:
            mean = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
            variance = tf.reduce_mean(tf.square(features - mean), axis=[1, 2])
            return tf.reduce_mean(variance, axis=-1, keepdims=True)
        mean = tf.reduce_mean(features, axis=1, keepdims=True)
        return tf.reduce_mean(tf.square(features - mean), axis=1, keepdims=True)

    def call(self, seg_base, seg_enhanced, clf_base, clf_enhanced, training=None):
        seg_disp = self.activation_dispersion(seg_enhanced)
        clf_disp = self.activation_dispersion(clf_enhanced)

        seg_norm = seg_disp / (tf.reduce_mean(seg_disp) + 1e-8)
        clf_norm = clf_disp / (tf.reduce_mean(clf_disp) + 1e-8)

        dispersion = tf.concat([seg_norm, clf_norm], axis=-1)
        adaptive_weights = self.weight_network(dispersion)

        seg_weight = tf.reshape(adaptive_weights[:, 0:1], [-1, 1, 1, 1])
        clf_weight = tf.reshape(adaptive_weights[:, 1:2], [-1, 1])

        seg_final = seg_base + seg_weight * (seg_enhanced - seg_base)
        clf_final = clf_base + clf_weight * (clf_enhanced - clf_base)

        return seg_final, clf_final

    def get_config(self):
        return super().get_config()
