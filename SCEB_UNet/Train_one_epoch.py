"""One training or validation epoch for segmentation models."""

from pathlib import Path
import time

import torch
import torch.nn.functional as F

import Config as config
from utils import iou_on_batch, save_on_batch


def make_boundary_target(masks, kernel_size):
    """Create a binary morphological boundary band from binary masks."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("boundary_kernel_size must be an odd integer >= 3")
    padding = kernel_size // 2
    masks = (masks > 0.5).float()
    dilation = F.max_pool2d(masks, kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-masks, kernel_size, stride=1, padding=padding)
    return ((dilation - erosion) > 0).float()


def boundary_loss(logits, targets, smooth=1e-5):
    """Balanced BCE plus soft Dice for the sparse boundary target."""
    positive = targets.sum().detach()
    negative = targets.numel() - positive
    positive_weight = (negative / (positive + smooth)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight
    )
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
    return 0.5 * (bce + dice)


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)


def print_summary(epoch, index, batches, loss, average_loss, average_time, dice,
                  average_dice, mode, learning_rate, logger):
    message = (
        f"   [{mode}] Epoch: [{epoch}][{index}/{batches}]  "
        f"Loss:{loss:.3f} (Avg {average_loss:.4f}) "
        f"Dice:{dice:.4f} (Avg {average_dice:.4f}) "
    )
    if mode == "Train":
        message += f"LR {learning_rate:.2e}   "
    message += f"(AvgTime {average_time:.2f})"
    logger.info(message)


def train_one_epoch(loader, model, criterion, optimizer, writer, epoch,
                    lr_scheduler, model_type, logger, device=None, scaler=None):
    """Run one epoch without retaining batch computation graphs in statistics."""
    del lr_scheduler, model_type  # Scheduler is stepped once per epoch by train_model.py.
    is_training = model.training
    mode = "Train" if is_training else "Val"
    device = device or next(model.parameters()).device
    stable = config.numerical_protocol == "stable_logits_v2"

    sample_count = 0
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    time_sum = 0.0
    segmentation_loss_sum = 0.0
    boundary_loss_sum = 0.0
    deep_loss_sum = 0.0
    optimizer_updates = 0
    optimizer_skips = 0
    zero_gradient_fractions = []
    gradient_norms = []
    saturated_pixels = 0
    prediction_pixels = 0

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    for index, (sampled_batch, names) in enumerate(loader, 1):
        start = time.time()
        images = sampled_batch["image"].to(device, non_blocking=True)
        masks = sampled_batch["label"].to(device, non_blocking=True).float()

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.bfloat16
                    if config.amp_dtype == "bfloat16"
                    else torch.float16
                ),
                enabled=config.use_amp and device.type == "cuda",
            ):
                if config.boundary_fusion:
                    predictions, auxiliary = (
                        model(images, return_aux=True, return_logits=True) if stable
                        else model(images, return_aux=True)
                    )
                else:
                    predictions = model(images, return_logits=True) if stable else model(images)
                    auxiliary = {}

            # The stable path never differentiates BCE through a low-precision
            # sigmoid. The legacy probability path remains for old runs.
            raw_output = predictions.float()
            predictions = torch.sigmoid(raw_output) if stable else raw_output
            if auxiliary:
                auxiliary["boundary_logits"] = auxiliary["boundary_logits"].float()
                auxiliary["coarse_logits"] = tuple(
                    logits.float() for logits in auxiliary["coarse_logits"]
                )

            segmentation_loss = (criterion.forward_logits(raw_output, masks) if stable
                                 else criterion(predictions, masks))
            auxiliary_boundary_loss = predictions.new_zeros(())
            auxiliary_deep_loss = predictions.new_zeros(())
            if auxiliary:
                boundary_targets = make_boundary_target(
                    masks, config.boundary_kernel_size
                )
                auxiliary_boundary_loss = boundary_loss(
                    auxiliary["boundary_logits"], boundary_targets
                )
                coarse_losses = []
                for coarse_logits in auxiliary["coarse_logits"]:
                    coarse_targets = F.interpolate(
                        masks, size=coarse_logits.shape[-2:], mode="nearest"
                    )
                    coarse_losses.append(
                        criterion.forward_logits(coarse_logits, coarse_targets) if stable
                        else criterion(torch.sigmoid(coarse_logits), coarse_targets)
                    )
                auxiliary_deep_loss = torch.stack(coarse_losses).mean()

            loss = (
                segmentation_loss
                + config.boundary_loss_weight * auxiliary_boundary_loss
                + config.deep_supervision_weight * auxiliary_deep_loss
            )
            if stable:
                if not torch.isfinite(loss.detach()).all():
                    raise FloatingPointError(f"Nonfinite {mode} loss at epoch {epoch+1}, batch {index}")
                saturated_pixels += int(((predictions.detach() == 0)
                                         | (predictions.detach() == 1)).sum())
                prediction_pixels += predictions.numel()
            if is_training:
                accumulation = config.gradient_accumulation_steps
                group_start = ((index - 1) // accumulation) * accumulation + 1
                group_end = min(group_start + accumulation - 1, len(loader))
                group_microbatches = group_end - group_start + 1
                backward_loss = loss / group_microbatches
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()

                if index % accumulation == 0 or index == len(loader):
                    if stable:
                        if scaler is not None and scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        gradients = [p.grad for p in model.parameters() if p.grad is not None]
                        norms = torch.stack(torch._foreach_norm(gradients))
                        if not torch.isfinite(norms).all():
                            raise FloatingPointError(
                                f"Nonfinite gradients at epoch {epoch+1}, batch {index}"
                            )
                        # Sample only the first convolution, keeping diagnostics cheap.
                        first_gradient = next(
                            (g for g in gradients if g.ndim >= 2), gradients[0]
                        ).detach().flatten()[:4096]
                        zero_gradient_fractions.append(float((first_gradient == 0).float().mean()))
                        gradient_norms.append(float(torch.linalg.vector_norm(norms)))
                    if scaler is not None and scaler.is_enabled():
                        previous_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        skipped = scaler.get_scale() < previous_scale
                        optimizer_skips += int(skipped)
                        optimizer_updates += int(not skipped)
                    else:
                        optimizer.step()
                        optimizer_updates += 1
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        batch_loss = _as_float(loss)
        batch_segmentation_loss = _as_float(segmentation_loss)
        batch_boundary_loss = _as_float(auxiliary_boundary_loss)
        batch_deep_loss = _as_float(auxiliary_deep_loss)
        batch_dice = _as_float(criterion._show_dice(predictions, masks))
        batch_iou = _as_float(iou_on_batch(masks.detach().clone(), predictions.detach().clone()))
        elapsed = time.time() - start
        if device.type == "cuda" and config.gpu_duty_cycle < 1.0:
            # Wait for this batch to finish before yielding GPU time. Sleep is
            # CPU-idle and does not alter batches, RNG, optimizer or scheduling.
            torch.cuda.synchronize(device)
            active_seconds = time.time() - start
            time.sleep(active_seconds * (1.0 / config.gpu_duty_cycle - 1.0))
            elapsed = time.time() - start

        sample_count += batch_size
        loss_sum += batch_size * batch_loss
        segmentation_loss_sum += batch_size * batch_segmentation_loss
        boundary_loss_sum += batch_size * batch_boundary_loss
        deep_loss_sum += batch_size * batch_deep_loss
        dice_sum += batch_size * batch_dice
        iou_sum += batch_size * batch_iou
        time_sum += batch_size * elapsed

        if not is_training and epoch % config.vis_frequency == 0:
            vis_path = Path(config.visualize_path) / str(epoch)
            vis_path.mkdir(parents=True, exist_ok=True)
            save_on_batch(images, masks, predictions, names, str(vis_path) + "/")

        if index % config.print_frequency == 0 or index == len(loader):
            print_summary(
                epoch + 1,
                index,
                len(loader),
                batch_loss,
                loss_sum / sample_count,
                time_sum / sample_count,
                batch_dice,
                dice_sum / sample_count,
                mode,
                min(group["lr"] for group in optimizer.param_groups),
                logger,
            )

    if sample_count == 0:
        raise RuntimeError(f"{mode} loader is empty")

    average_loss = loss_sum / sample_count
    average_dice = dice_sum / sample_count
    average_iou = iou_sum / sample_count
    if stable:
        logger.info(
            f"Numerical health [{mode}] epoch={epoch+1} updates={optimizer_updates} "
            f"skips={optimizer_skips} "
            f"first_conv_zero_grad_fraction={sum(zero_gradient_fractions)/max(len(zero_gradient_fractions),1):.6g} "
            f"mean_grad_norm={sum(gradient_norms)/max(len(gradient_norms),1):.6g} "
            f"probability_saturation={saturated_pixels/max(prediction_pixels,1):.6g}"
        )
    if writer is not None:
        writer.add_scalar(f"epoch_{mode}_loss", average_loss, epoch)
        writer.add_scalar(f"epoch_{mode}_iou", average_iou, epoch)
        writer.add_scalar(f"epoch_{mode}_dice", average_dice, epoch)
        writer.add_scalar(
            f"epoch_{mode}_segmentation_loss",
            segmentation_loss_sum / sample_count,
            epoch,
        )
        if config.boundary_fusion:
            writer.add_scalar(
                f"epoch_{mode}_boundary_loss",
                boundary_loss_sum / sample_count,
                epoch,
            )
            writer.add_scalar(
                f"epoch_{mode}_deep_supervision_loss",
                deep_loss_sum / sample_count,
                epoch,
            )

    return average_loss, average_dice
