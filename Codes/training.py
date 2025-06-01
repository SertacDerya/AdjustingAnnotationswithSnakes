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

            if batch_idx == 0 and iterations % 10 == 0:
                img_to_plot = utils.from_torch(images[0])
                lbl_to_plot = utils.from_torch(labels[0])
                prd_to_plot = utils.from_torch(preds[0]) 

                if img_to_plot.ndim == 4:
                    img_to_plot = img_to_plot[0]
                if lbl_to_plot.ndim == 4:
                    lbl_to_plot = lbl_to_plot[0]
                if prd_to_plot.ndim == 4:
                    prd_to_plot = prd_to_plot[0]
                
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
                    slice_x = np.min(data_vol, axis=0)
                    im_x = axes[i, 0].imshow(slice_x.T, cmap=cmap, origin='lower')
                    axes[i, 0].set_title(f"{title_prefix} (Y-Z projection)")
                    fig.colorbar(im_x, ax=axes[i, 0], orientation='horizontal', fraction=0.046, pad=0.04)
                    axes[i, 0].axis('off')

                    slice_y = np.min(data_vol, axis=1)
                    im_y = axes[i, 1].imshow(slice_y.T, cmap=cmap, origin='lower')
                    axes[i, 1].set_title(f"{title_prefix} (X-Z projection)")
                    fig.colorbar(im_y, ax=axes[i, 1], orientation='horizontal', fraction=0.046, pad=0.04)
                    axes[i, 1].axis('off')

                    slice_z = np.min(data_vol, axis=2)
                    im_z = axes[i, 2].imshow(slice_z.T, cmap=cmap, origin='lower') # Transpose for consistent view with X,Y
                    axes[i, 2].set_title(f"{title_prefix} (X-Y projection)")
                    fig.colorbar(im_z, ax=axes[i, 2], orientation='horizontal', fraction=0.046, pad=0.04)
                    axes[i, 2].axis('off')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                plot_filename = os.path.join("./trainplot/", f"plot_iter_{iterations}_batch_{batch_idx}.png")
                plt.savefig(plot_filename)
                plt.close(fig)
                logger.info(f"Saved visualization to {plot_filename}")

        return {"loss": float(mean_loss/len(self.dataloader))}

def show_slices(distance_map, graph):
    """Show orthogonal slices through the distance map"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    xx = np.min(distance_map, axis=0)
    yy = np.min(distance_map, axis=1)
    zz = np.min(distance_map, axis=2)
    
    im0 = axes[0].imshow(xx.T, cmap='coolwarm')
    axes[0].set_title('Slice (X)')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(yy.T, cmap='coolwarm')
    axes[1].set_title('Slice (Y)')
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(zz.T, cmap='coolwarm')
    axes[2].set_title('Slice (Z)')
    fig.colorbar(im2, ax=axes[2])

    for edge in graph.edges:
        node1, node2 = edge
        x1, y1, z1 = graph.nodes[node1]['pos']
        x2, y2, z2 = graph.nodes[node2]['pos']
        
        axes[0].plot([y1, y2], [z1, z2], 'r-')
        axes[1].plot([x1, x2], [z1, z2], 'r-')
        axes[2].plot([x1, x2], [y1, y2], 'r-')
    
    plt.show() 
    
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

        utils.mkdir("./val_plot/")

    def __call__(self, iteration, network, loss_function):

        losses = []
        preds_collector = []
        scores = {"corr":[], "comp":[], "qual":[]}

        drive_output_path = "/content/drive/MyDrive/snake_model_outputs"
        plot_save_path = "./val_plot/"

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

                pred_np_item = utils.from_torch(pred.cpu())[0]
                label_np_item = utils.from_torch(label.cpu())[0]
                image_np_item = utils.from_torch(image.cpu())[0]
                
                preds_collector.append(pred_np_item)
                
                pred_mask_item = skeletonize_3d((pred_np_item[0] < 5))//255
                label_object_mask_item = (label_np_item[0] == 0)
                label_skeleton_item = skeletonize_3d(label_object_mask_item)//255

                corr, comp, qual = correctness_completeness_quality(pred_mask_item, label_object_mask_item, slack=3)
                
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                fig, axes = plt.subplots(3, 3, figsize=(15, 16))
                fig.suptitle(f"Validation: Iteration {iteration} - Sample {i}\nLoss: {loss_v:.4f} | Qual: {qual:.3f}", fontsize=14)

                plot_data_config = [
                    (image_np_item[0], "Input Image", 'gray', None),
                    (label_np_item[0], "Label", 'viridis', label_skeleton_item),
                    (pred_np_item[0], "Prediction", 'viridis', pred_mask_item)
                ]
                
                projection_titles = ["(Y-Z Projection)", "(X-Z Projection)", "(X-Y Projection)"]
                projection_axes = [0, 1, 2]

                for row_idx, (data_vol, title_prefix, cmap, skeleton_vol) in enumerate(plot_data_config):
                    for col_idx, proj_axis in enumerate(projection_axes):
                        ax = axes[row_idx, col_idx]
                        
                        slice_proj = np.min(data_vol, axis=proj_axis)
                        im = ax.imshow(slice_proj.T, cmap=cmap, origin='lower', aspect='auto')
                        
                        plot_title = f"{title_prefix} {projection_titles[col_idx]}"
                        if skeleton_vol is not None:
                            skeleton_slice_proj = np.max(skeleton_vol, axis=proj_axis) 
                            ax.contour(skeleton_slice_proj.T, colors='red', linewidths=0.8, levels=[0.5], origin='lower')
                            plot_title += " + Skel"

                        ax.set_title(plot_title, fontsize=10)
                        ax.axis('off')
                        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)

                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                
                plot_filename = os.path.join(plot_save_path, f"val_plot_iter_{iteration}_sample_{i}.png")
                plt.savefig(plot_filename)
                plt.close(fig)
                # logger.info(f"Saved validation visualization to {plot_filename}")

                output_valid_dir = os.path.join(self.output_path, "output_valid_predictions")
                utils.mkdir(output_valid_dir)
                np.save(os.path.join(output_valid_dir, "pred_iter{:04d}_sample{:03d}.npy".format(iteration, i)), pred_np_item)

        scores["qual"] = np.nan_to_num(scores["qual"])
        
        qual_total = np.mean(scores["qual"],axis=0)
        corr_total = np.mean(scores["corr"],axis=0)
        comp_total = np.mean(scores["comp"],axis=0)

        if self.bestqual < qual_total:
            self.bestqual = qual_total
            for i in range(len(self.dataloader_val)):
                np.save(os.path.join(output_valid_dir, "pred_{:06d}_bestqual.npy".format(i,iteration)), preds_collector[i])
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
    
    
    