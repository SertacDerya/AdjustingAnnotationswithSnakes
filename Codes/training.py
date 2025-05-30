import time
import numpy as np
import os
import logging
import torch
from . import utils
from skimage.morphology import skeletonize
from .scores import correctness_completeness_quality
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Trainer(object):

    def __init__(self,training_step, validation=None, valid_every=None,
                 print_every=None, save_every=None, save_path=None, save_objects={},
                 save_callback=None):

        self.training_step = training_step
        self.validation = validation
        self.valid_every = valid_every
        self.print_every = print_every
        self.save_every = save_every
        self.save_path = save_path
        self.save_objects = save_objects
        self.save_callback = save_callback

        self.starting_iteration = 0

        if self.save_path is None or self.save_path in ["", ".", "./"]:
            self.save_path = os.getcwd()

        self.results = {"training":{"epochs":[], "results":[]},
                        "validation": {"epochs":[], "results":[]}}

    def save_state(self, iteration):

        for name, object in self.save_objects.items():
            utils.torch_save(os.path.join(self.save_path, "{}_final.pickle".format(name, iteration)),
                             object.state_dict())

        utils.pickle_write(os.path.join(self.save_path, "results_final.pickle".format(iteration)),
                           self.results)

        if self.save_callback is not None:
            self.save_callback(iteration)

    def run_for(self, iterations):

        start_time = time.time()
        block_iter_start_time = time.time()

        for iteration in range(self.starting_iteration, self.starting_iteration+iterations+1):

            train_step_results = self.training_step(iteration)
            self.results['training']["results"].append(train_step_results)
            self.results['training']["epochs"].append(iteration)

            if self.print_every is not None:
                if iteration%self.print_every==0:
                    elapsed_time = (time.time() - start_time)//60
                    block_iter_elapsed_time = time.time() - block_iter_start_time

                    loss_v1 = train_step_results["loss"] if "loss" in train_step_results.keys() else None
                    loss_v2 = train_step_results["loss_2"] if "loss_2" in train_step_results.keys() else 0
                    to_print = "[{:0.0f}min][{:0.2f}s] - Epoch: {} (Train-batch Loss: {:0.6f}, {:0.6f})"
                    to_print = to_print.format(elapsed_time, block_iter_elapsed_time, iteration, loss_v1, loss_v2)

                    logger.info(to_print)
                    block_iter_start_time = time.time()

            if self.validation is not None and self.valid_every is not None:
                if iteration%self.valid_every==0 and iteration!=self.starting_iteration:
                    logger.info("validation...")
                    start_valid = time.time()

                    validation_results = self.validation(iteration)

                    self.results['validation']["results"].append(validation_results)
                    self.results['validation']["epochs"].append(iteration)
                    logger.info("Validation time: {:.2f}s".format(time.time()-start_valid))

            if self.save_every is not None:
                if iteration%self.save_every==0 and iteration!=self.starting_iteration:
                    start_saving = time.time()
                    self.save_state(iteration)
                    logger.info("Saving time: {:.2f}s".format(time.time()-start_saving))

class TrainingEpoch(object):

    def __init__(self, dataloader, ours=False, ours_start=0):

        self.dataloader = dataloader
        self.ours = ours
        self.ours_start = ours_start
        utils.mkdir("./trainplot/")

    def __call__(self, iterations, network, optimizer, lr_scheduler, base_loss, our_loss):
        
        mean_loss = 0
        for images, labels, masks, graphs, slices in self.dataloader:

            images = images.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            
            preds = network(images.contiguous())
            # apply the mask. might create confusion for the unet
            #binary_mask = (masks == 0).float() 
            #preds = preds * binary_mask

            if iterations % 50 == 0:
                # Plot and save the first image, label, and prediction of the first batch
                try:
                    img_to_plot = utils.from_torch(images[0]) # (C, H, W)
                    lbl_to_plot = utils.from_torch(labels[0]) # (C_label, H, W)
                    prd_to_plot = utils.from_torch(preds[0])  # (C_out, H, W)

                    # Prepare for plotting (e.g., select first channel, transpose if necessary)
                    # Assuming single-channel or taking the first channel for plotting
                    if img_to_plot.shape[0] > 1: # More than one channel
                        if img_to_plot.shape[0] == 3: # RGB
                             img_to_plot = img_to_plot.transpose(1, 2, 0) # H, W, C
                        else: # Grayscale with multiple channels, take first
                            img_to_plot = img_to_plot[0] # H, W
                    else: # Single channel
                        img_to_plot = img_to_plot.squeeze(0) # H, W
                    
                    lbl_to_plot = lbl_to_plot.squeeze(0) # H, W (assuming single channel label)
                    prd_to_plot = prd_to_plot.squeeze(0) # H, W (assuming single channel prediction or taking first)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    fig.suptitle(f"Epoch {iterations} - Batch {i}")

                    axes[0].imshow(img_to_plot, cmap='gray')
                    axes[0].set_title("Input Image")
                    axes[0].axis('off')

                    axes[1].imshow(lbl_to_plot, cmap='viridis') # Or another cmap suitable for distance maps
                    axes[1].set_title("Ground Truth Label")
                    axes[1].axis('off')

                    axes[2].imshow(prd_to_plot, cmap='viridis') # Or another cmap suitable for distance maps
                    axes[2].set_title("Network Prediction")
                    axes[2].axis('off')

                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                    plot_filename = os.path.join("./trainplot/", f"epoch_{iterations}_batch_{i}_visualization.png")
                    plt.savefig(plot_filename)
                    plt.close(fig)
                    logger.info(f"Saved visualization to {plot_filename}")
                except Exception as e:
                    logger.error(f"Error during plotting/saving visualization: {e}")
                
            if self.ours and iterations >= self.ours_start:
            # calls forward on loss here, and snake is adjusted
                loss = our_loss(preds, graphs, slices, None, iterations)
            else:
                loss = base_loss(preds, labels)
                
            loss_v = float(utils.from_torch(loss))

            if np.isnan(loss_v) or np.isinf(loss_v):
                return {"loss": loss_v,
                        "pred": utils.from_torch(preds),
                        "labels": utils.from_torch(labels)}
            
            mean_loss += loss_v
            optimizer.zero_grad()
            loss.backward()
            # optimizer optimizes the network parameters
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if torch.cuda.is_available() and iterations % 10 == 0:
                torch.cuda.empty_cache()

        return {"loss": float(mean_loss/len(self.dataloader))}
    
    
class Validation(object):

    def __init__(self, crop_size, margin_size, dataloader_val, out_channels, output_path):
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.dataloader_val = dataloader_val
        self.out_channels = out_channels
        self.output_path = output_path

        self.bestqual = 0
        self.bestcomp = 0
        self.bestcorr = 0

    def __call__(self, iteration, network, loss_function):

        losses = []
        preds = []
        scores = {"corr":[], "comp":[], "qual":[]}

        drive_output_path = "/content/drive/MyDrive/snake_model_outputs"
        local_output_path = self.output_path
        drive_valid_path = os.path.join(drive_output_path, "output_valid")
        local_valid_path = os.path.join(local_output_path, "output_valid")
        utils.mkdir(drive_valid_path)
        utils.mkdir(local_valid_path)

        network.train(False)
        with utils.torch_no_grad:
            for i, data_batch in enumerate(self.dataloader_val):
                image, label, mask, graph = data_batch
                image  = image.cuda()
                label  = label.cuda()
                mask   = mask.cuda()

                out_shape = (image.shape[0],self.out_channels,*image.shape[2:])
                pred = utils.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred = utils.process_in_chuncks(image, pred,
                                            lambda chunk: network(chunk),
                                            self.crop_size, self.margin_size)
                # apply the mask again
                #binary_mask = (mask == 0).float()
                #pred = pred * binary_mask

                loss = loss_function(pred, label)
                loss_v = float(utils.from_torch(loss))
                losses.append(loss_v)

                pred_np = utils.from_torch(pred)[0]
                preds.append(pred_np)
                label_np = utils.from_torch(label)[0]
                
                pred_mask = skeletonize((pred_np <= 5)[0])//255
                label_mask = (label_np==0)

                corr, comp, qual = correctness_completeness_quality(pred_mask, label_mask, slack=3)
                
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                # Save input and ground truth (only once)
                input_filename_local = os.path.join(local_valid_path, "val_input_{:03d}.npy".format(i))
                if not os.path.exists(input_filename_local):
                    np.save(input_filename_local, utils.from_torch(image)[0])

                gt_filename_local = os.path.join(local_valid_path, "val_gt_{:03d}.npy".format(i))
                if not os.path.exists(gt_filename_local):
                    np.save(gt_filename_local, label_np)

                # Save predictions to BOTH Google Drive and local path
                # 1. Save to Google Drive
                drive_pred_filename = os.path.join(drive_valid_path, "val_pred_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(drive_pred_filename, pred_np)
                
                drive_pred_mask_filename = os.path.join(drive_valid_path, "val_predmask_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(drive_pred_mask_filename, pred_mask)
                
                # 2. Save to local path (the one you're missing)
                local_pred_filename = os.path.join(local_valid_path, "val_pred_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(local_pred_filename, pred_np)
                
                local_pred_mask_filename = os.path.join(local_valid_path, "val_predmask_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(local_pred_mask_filename, pred_mask)

        scores["qual"] = np.nan_to_num(scores["qual"])
        
        qual_total = np.mean(scores["qual"],axis=0)
        corr_total = np.mean(scores["corr"],axis=0)
        comp_total = np.mean(scores["comp"],axis=0)

        # Also save best quality model
        if self.bestqual < qual_total:
            self.bestqual = qual_total
            for i in range(len(self.dataloader_val)):
                # Save to local path (as before)
                np.save(os.path.join(local_valid_path, "pred_{:06d}_bestqual.npy".format(i)), preds[i])
            utils.torch_save(os.path.join(self.output_path, "network_bestqual.pickle"),
                             network.state_dict())
        
        # Save metrics to a CSV file for tracking
        metrics_file = os.path.join(local_output_path, "metrics.csv")
        # Create file with header if it doesn't exist
        if not os.path.exists(metrics_file):
            with open(metrics_file, 'w') as f:
                f.write("Iteration,Loss,Correctness,Completeness,Quality\n")
        
        # Append metrics
        with open(metrics_file, 'a') as f:
            f.write(f"{iteration},{np.mean(losses):.6f},{corr_total:.6f},{comp_total:.6f},{qual_total:.6f}\n")

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean qual: {:0.3f}".format(qual_total))
        logger.info("\tMean corr: {:0.3f}".format(corr_total))
        logger.info("\tMean comp: {:0.3f}".format(comp_total))
        logger.info("Best quality score is {}".format(self.bestqual))
        
        network.train(True)

        return {"loss": np.mean(losses),
                "mean_corr": corr_total,
                "mean_comp": comp_total,
                "mean_qual": qual_total,
                "scores": scores}
    
    
    
