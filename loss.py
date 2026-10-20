import tensorflow as tf


# ============================================================================
# GEOMETRIC FILTERS
# ============================================================================

class GeometricFilters:
    @staticmethod
    def curvature_filter(kernel_size=5, eps=1e-8):
        k   = int(kernel_size)
        x   = tf.cast(tf.range(k) - (k // 2), tf.float32)
        y   = tf.cast(tf.range(k) - (k // 2), tf.float32)
        xx, yy = tf.meshgrid(x, y)
        r2  = xx * xx + yy * yy
        # sigma = k/4, so the 5x5 kernel used here gives sigma = 1.25.
        sigma   = tf.cast(k / 4.0, tf.float32)
        gaussian = tf.exp(-r2 / (2.0 * (sigma * sigma + eps)))
        f   = (xx * xx - yy * yy) * gaussian
        f   = f / (tf.reduce_sum(tf.abs(f)) + eps)
        return tf.reshape(f, [k, k, 1, 1])


# ============================================================================
# SEGMENTATION LOSS
# ============================================================================

def stabilized_tversky_index(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-7):
    """Per-image Tversky index. Returns shape [B].

    WARNING: enhanced_lesion_focus_loss() below calls this WITHOUT alpha or
    beta, so these symmetric defaults apply and the term reduces to a focal
    DICE loss. The asymmetric alpha=0.3 / beta=0.7 penalty is not applied
    anywhere in this repository. If the asymmetric form is intended, pass it
    explicitly at the call site and re-run every reported result.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    axes = [1, 2, 3]
    tp = tf.reduce_sum(y_true * y_pred,            axis=axes)
    fp = tf.reduce_sum((1 - y_true) * y_pred,      axis=axes)
    fn = tf.reduce_sum(y_true * (1 - y_pred),      axis=axes)
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def stabilized_focal_tversky(y_true, y_pred, gamma=0.75):
    """Focal Tversky loss (scalar)."""
    eps   = 1e-7
    t_idx = stabilized_tversky_index(y_true, y_pred, smooth=eps)
    focal = tf.pow(tf.maximum(1.0 - t_idx, eps), gamma)
    focal = tf.where(tf.math.is_finite(focal), focal, tf.ones_like(focal))
    return tf.reduce_mean(focal)


def efficient_boundary_detection(y_true, y_pred, kernel_size=5):
    """Curvature-based boundary loss (scalar)."""
    eps    = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    # NOTE: y_pred is hard-thresholded here, so this term is NOT
    # differentiable with respect to the mask probabilities. It changes the
    # logged loss value only (selection monitors val_combined), and supplies no
    # boundary-training gradient. Treat it as a diagnostic, not a penalty.
    y_true_bin = tf.clip_by_value(tf.cast(y_true > 0.5, tf.float32), 0.0, 1.0)
    y_pred_bin = tf.clip_by_value(tf.cast(y_pred > 0.5, tf.float32), 0.0, 1.0)

    kernel = GeometricFilters.curvature_filter(kernel_size)
    b_true = tf.nn.conv2d(y_true_bin, kernel, strides=1, padding='SAME')
    b_pred = tf.nn.conv2d(y_pred_bin, kernel, strides=1, padding='SAME')

    per_sample = tf.clip_by_value(
        tf.reduce_mean(tf.abs(b_true - b_pred), axis=[1, 2, 3]), 0.0, 1.0
    )
    per_sample = tf.where(tf.math.is_finite(per_sample), per_sample, tf.zeros_like(per_sample))
    return tf.reduce_mean(per_sample)


def efficient_texture_consistency(y_true, y_pred):
    """Horizontal mask-gradient dispersion penalty (scalar).

    Sobel is applied to the MASKS (y_true, y_pred), not to the ultrasound
    image, so this does not measure image texture. The paper names it
    accordingly.
    """
    eps    = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])

    gt = tf.nn.conv2d(y_true, sobel_x, strides=1, padding='SAME')
    gp = tf.nn.conv2d(y_pred, sobel_x, strides=1, padding='SAME')

    axes = [1, 2, 3]
    std_gt = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(gt - tf.reduce_mean(gt, axis=axes, keepdims=True)), axis=axes), 0.0) + eps)
    std_gp = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(gp - tf.reduce_mean(gp, axis=axes, keepdims=True)), axis=axes), 0.0) + eps)

    per_sample = tf.abs(std_gt - std_gp)
    per_sample = tf.where(tf.math.is_finite(per_sample), per_sample, tf.zeros_like(per_sample))
    return tf.reduce_mean(per_sample)


def enhanced_lesion_focus_loss(y_true, y_pred):
    """
    Combined segmentation loss:
        L_seg = L_FocalTversky + 0.25 * L_boundary + 0.15 * L_texture
    with per-sample lesion-size boost.
    """
    eps    = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(
        tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred)), eps, 1.0 - eps
    )

    # Per-sample focal Tversky
    t_idx   = stabilized_tversky_index(y_true, y_pred, smooth=eps)
    base    = tf.pow(tf.maximum(1.0 - t_idx, eps), 0.75)

    # Boundary + texture (scalars broadcast to [B])
    b_loss = efficient_boundary_detection(y_true, y_pred)
    t_loss = efficient_texture_consistency(y_true, y_pred)
    b_vec  = tf.fill(tf.shape(base), tf.cast(b_loss, tf.float32))
    t_vec  = tf.fill(tf.shape(base), tf.cast(t_loss, tf.float32))

    # Lesion-size boost: small lesions get higher weight
    axes         = [1, 2, 3]
    total_pixels = tf.cast(tf.shape(y_true)[1] * tf.shape(y_true)[2] * tf.shape(y_true)[3], tf.float32)
    lesion_ratio = tf.clip_by_value(tf.reduce_sum(y_true, axis=axes) / tf.maximum(total_pixels, 1.0), 0.0, 1.0)
    # Factor lies in [1, 2] and attains its MAXIMUM at lesion_ratio = 0, i.e.
    # on EMPTY masks, not on small lesions. BUSI normal cases are therefore
    # upweighted most unless excluded upstream.
    size_boost   = 1.0 + tf.clip_by_value(tf.exp(-10.0 * lesion_ratio), 0.0, 2.0)

    loss = tf.clip_by_value(base * size_boost + 0.25 * b_vec + 0.15 * t_vec, 0.0, 10.0)
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
    return tf.reduce_mean(loss)


# ============================================================================
# CLASSIFICATION LOSS
# ============================================================================

def enhanced_multi_modal_focal_loss(y_true_clf, y_pred_clf):
    """Focal cross-entropy for multi-class classification (scalar)."""
    eps    = 1e-7
    y_pred = tf.cast(y_pred_clf, tf.float32)
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.fill(tf.shape(y_pred), eps))
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Re-normalise
    denom  = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = y_pred / tf.where(denom > 0.0, denom, tf.ones_like(denom))
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    y_true = tf.cast(tf.reshape(y_true_clf, [-1]), tf.int32)
    ce     = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    ce     = tf.where(tf.math.is_finite(ce), ce, tf.fill(tf.shape(ce), 1e3))

    pt     = tf.exp(-ce)
    focal  = tf.pow(tf.maximum(1.0 - pt, eps), 2.0) * ce
    return tf.reduce_mean(tf.clip_by_value(focal, 0.0, 10.0))
