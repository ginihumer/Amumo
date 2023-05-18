import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.cuda.amp import autocast
from hflayers import Hopfield

from .methods import cloob, clip
from .zero_shot import zero_shot_eval


def is_master(args):
    return (not args.distributed) or args.gpu == 0


def get_loss(model, images, texts, loss_fct_img, loss_fct_txt, hopfield_layer, args):
    image_features, text_features, inv_tau = model(images, texts)
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1:]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1:]
        )
    else:
        all_image_features = image_features
        all_text_features = text_features

    if args.method == "cloob":
        loss = cloob(
            all_image_features, all_text_features, inv_tau, hopfield_layer)
    elif args.method == "clip":
        loss = clip(
            all_image_features, all_text_features, inv_tau, loss_fct_img, loss_fct_txt, args)
    return loss


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)

    model.train()

    dataloader, sampler = data['train'].dataloader, data['train'].sampler

    loss_fct_img = nn.CrossEntropyLoss()
    loss_fct_txt = nn.CrossEntropyLoss()

    model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    with open(model_config_file, 'r') as f:
            model_info = json.load(f)
    hopfield_layer = Hopfield(input_size=model_info['embed_dim'], 
                              scaling=args.scale_hopfield, 
                              normalize_hopfield_space=False,
                              normalize_hopfield_space_affine=False,
                              normalize_pattern_projection=False,
                              normalize_pattern_projection_affine=False, 
                              normalize_state_pattern=False, 
                              normalize_state_pattern_affine=False, 
                              normalize_stored_pattern=False, 
                              normalize_stored_pattern_affine=False,
                              state_pattern_as_static=True,
                              pattern_projection_as_static=True,
                              stored_pattern_as_static=True,
                              disable_out_projection=True,
                              num_heads = 1,
                              dropout=False)

    if args.gpu is not None:
        loss_fct_img = loss_fct_img.cuda(args.gpu)
        loss_fct_txt = loss_fct_txt.cuda(args.gpu)
        hopfield_layer = hopfield_layer.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, texts, loss_fct_img, loss_fct_txt, hopfield_layer, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss(model, images, texts, loss_fct_img, loss_fct_txt, hopfield_layer, args)
            total_loss.backward()
            optimizer.step()

        scheduler.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_inv_tau.data = torch.clamp(m.logit_inv_tau.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            log_str = f""
            log_data = {}

            # logging
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tinv_tau {m.logit_inv_tau.data.exp():.3f}{log_str}"
            )

            # save train loss / etc.
            log_data.update({
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "inv_tau": m.logit_inv_tau.data.exp().item(),
                "lr": optimizer.param_groups[0]["lr"]
            })

            # log to tensorboard and/or wandb
            timestep = epoch * num_batches_per_epoch + i
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return

    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    dataloader = data['val'].dataloader

    loss_fct_img = nn.CrossEntropyLoss()
    loss_fct_txt = nn.CrossEntropyLoss()
    model_config_file = Path(__file__).parent / f"model_configs/{args.model.replace('/', '-')}.json"
    with open(model_config_file, 'r') as f:
            model_info = json.load(f)
    hopfield_layer = Hopfield(input_size=model_info['embed_dim'],
                              scaling=args.scale_hopfield, 
                              normalize_hopfield_space=False,
                              normalize_hopfield_space_affine=False,
                              normalize_pattern_projection=False,
                              normalize_pattern_projection_affine=False, 
                              normalize_state_pattern=False, 
                              normalize_state_pattern_affine=False, 
                              normalize_stored_pattern=False, 
                              normalize_stored_pattern_affine=False,
                              state_pattern_as_static=True,
                              pattern_projection_as_static=True,
                              stored_pattern_as_static=True,
                              disable_out_projection=True,
                              num_heads = 1,
                              dropout=False)
    if args.gpu is not None:
        loss_fct_img = loss_fct_img.cuda(args.gpu)
        loss_fct_txt = loss_fct_txt.cuda(args.gpu)
        hopfield_layer = hopfield_layer.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, texts = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, inv_tau = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

            # Calculate the loss
            if args.method == "cloob":
                total_loss = cloob(
                    image_features, text_features, inv_tau, hopfield_layer)
            elif args.method == "clip":
                total_loss = clip(
                    image_features, text_features, inv_tau, loss_fct_img, loss_fct_txt, args)

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = get_metrics(
            torch.cat(all_image_features), torch.cat(all_text_features)
        )
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features):
    metrics = {}
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = (
        torch.arange(len(text_features)).view(-1, 1).to(logits_per_image.device)
    )

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
