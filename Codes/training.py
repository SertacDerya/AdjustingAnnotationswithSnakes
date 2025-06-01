import time
import numpy as np
import os
import logging
from . import utils
from skimage.morphology import skeletonize_3d
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
        for batch_idx, (images, labels, graphs, slices) in enumerate(self.dataloader):

            images = images.cuda()
            labels = labels.cuda()
            
            preds = network(images.contiguous())
            
            if self.ours and iterations >= self.ours_start:
                loss = our_loss(preds, graphs, slices, iterations)
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
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if batch_idx == 0 and iterations % 10 == 0: # Adjust plotting frequency as needed
                img_to_plot = utils.from_torch(images[0]) # (C, D, H, W) or (D, H, W)
                lbl_to_plot = utils.from_torch(labels[0]) # (C_label, D, H, W) or (D, H, W)
                prd_to_plot = utils.from_torch(preds[0])  # (C_out, D, H, W) or (D, H, W)

                # Remove channel dim if it's 1, otherwise take the first channel
                if img_to_plot.ndim == 4: # C, D, H, W
                    img_to_plot = img_to_plot[0] # D, H, W
                if lbl_to_plot.ndim == 4:
                    lbl_to_plot = lbl_to_plot[0]
                if prd_to_plot.ndim == 4:
                    prd_to_plot = prd_to_plot[0]
                
                # Ensure they are 3D
                if not (img_to_plot.ndim == 3 and lbl_to_plot.ndim == 3 and prd_to_plot.ndim == 3):
                    print(f"Skipping plotting for iteration {iterations}, batch {batch_idx} due to unexpected dimensions.")
                    print(f"Image shape: {img_to_plot.shape}, Label shape: {lbl_to_plot.shape}, Pred shape: {prd_to_plot.shape}")
                    continue


                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                fig.suptitle(f"Iteration {iterations} - Batch {batch_idx} - Item 0 Slices")

                data_to_plot = [
                    (img_to_plot, "Input Image", 'gray'),
                    (lbl_to_plot, "Ground Truth Label", 'viridis'),
                    (prd_to_plot, "Network Prediction", 'viridis')
                ]

                for i, (data_vol, title_prefix, cmap) in enumerate(data_to_plot):
                    # X slice (min along axis 0 - Depth)
                    slice_x = np.min(data_vol, axis=0)
                    im_x = axes[i, 0].imshow(slice_x.T, cmap=cmap, origin='lower') # Transpose for consistent view with Y,Z
                    axes[i, 0].set_title(f"{title_prefix} (Y-Z projection)")
                    fig.colorbar(im_x, ax=axes[i, 0], orientation='horizontal', fraction=0.046, pad=0.04)
                    axes[i, 0].axis('off')

                    # Y slice (min along axis 1 - Height)
                    slice_y = np.min(data_vol, axis=1)
                    im_y = axes[i, 1].imshow(slice_y.T, cmap=cmap, origin='lower') # Transpose for consistent view with X,Z
                    axes[i, 1].set_title(f"{title_prefix} (X-Z projection)")
                    fig.colorbar(im_y, ax=axes[i, 1], orientation='horizontal', fraction=0.046, pad=0.04)
                    axes[i, 1].axis('off')

                    # Z slice (min along axis 2 - Width)
                    slice_z = np.min(data_vol, axis=2)
                    im_z = axes[i, 2].imshow(slice_z.T, cmap=cmap, origin='lower') # Transpose for consistent view with X,Y
                    axes[i, 2].set_title(f"{title_prefix} (X-Y projection)")
                    fig.colorbar(im_z, ax=axes[i, 2], orientation='horizontal', fraction=0.046, pad=0.04)
                    axes[i, 2].axis('off')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
                
                # Save or show the plot
                plot_filename = os.path.join("./trainplot/", f"plot_iter_{iterations}_batch_{batch_idx}.png")
                plt.savefig(plot_filename)
                plt.close(fig)
                logger.info(f"Saved visualization to {plot_filename}")

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

        network.train(False)
        with utils.torch_no_grad:
            for i, (image, label) in enumerate(self.dataloader_val):
        
                image  = image.cuda()[:,None]
                label  = label.cuda()[:,None]

                out_shape = (image.shape[0],self.out_channels,*image.shape[2:])
                pred = utils.to_torch(np.empty(out_shape, np.float32), volatile=True).cuda()
                pred = utils.process_in_chuncks(image, pred,
                                            lambda chunk: network(chunk),
                                            self.crop_size, self.margin_size)

                loss = loss_function(pred, label)
                loss_v = float(utils.from_torch(loss))
                losses.append(loss_v)

                pred_np = utils.from_torch(pred)[0]
                preds.append(pred_np)
                label_np = utils.from_torch(label)[0]
                
                pred_mask = skeletonize_3d((pred_np < 5)[0])//255
                label_mask = (label_np==0)

                corr, comp, qual = correctness_completeness_quality(pred_mask, label_mask, slack=3)
                
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                # save the prediction here
                output_valid = os.path.join(self.output_path, "output_valid")
                utils.mkdir(output_valid)
                np.save(os.path.join(output_valid, "pred_{:06d}_final.npy".format(i,iteration)), pred_np)

        scores["qual"] = np.nan_to_num(scores["qual"])
        
        qual_total = np.mean(scores["qual"],axis=0)
        corr_total = np.mean(scores["corr"],axis=0)
        comp_total = np.mean(scores["comp"],axis=0)

        if self.bestqual < qual_total:
            self.bestqual = qual_total
            for i in range(len(self.dataloader_val)):
                np.save(os.path.join(output_valid, "pred_{:06d}_bestqual.npy".format(i,iteration)), preds[i])
            utils.torch_save(os.path.join(self.output_path, "network_bestqual.pickle"),
                             network.state_dict())
        

        logger.info("\tMean loss: {}".format(np.mean(losses)))
        logger.info("\tMean qual: {:0.3f}".format(qual_total))
        logger.info("Best quality score is {}".format(self.bestqual))
        
        network.train(True)

        return {"loss": np.mean(losses),
                "mean_corr": corr_total,
                "mean_comp": comp_total,
                "mean_qual": qual_total,
                "scores": scores}
    
    
    