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
        for idx, (images, labels, masks, graphs, slices) in enumerate(self.dataloader):

            images = images.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            
            preds = network(images.contiguous())
            # apply the mask. might create confusion for the unet
            #binary_mask = (masks == 0).float() 
            #preds = preds * binary_mask
            snake = None
            if self.ours and iterations >= self.ours_start:
            # calls forward on loss here, and snake is adjusted
                loss, snake = our_loss(preds, graphs, slices, iterations)
            else:
                loss = base_loss(preds, labels)

            if iterations % 50 == 0:
                # Plot and save the first image, label, and prediction of the first batch
                try:
                    img_to_plot_orig = utils.from_torch(images[0]/255.0) # (C, H, W)
                    lbl_to_plot = utils.from_torch(labels[0]) # (C_label, H, W)
                    prd_to_plot = utils.from_torch(preds[0])  # (C_out, H, W)

                    # Prepare for plotting (e.g., select first channel, transpose if necessary)
                    img_to_plot = img_to_plot_orig.copy()
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
                    fig.suptitle(f"Epoch {iterations} - Batch")

                    # Plot Input Image (with potential snake overlay)
                    axes[0].imshow(img_to_plot, cmap='gray' if img_to_plot.ndim == 2 else None)
                    axes[0].set_title("Input Image")
                    if snake and hasattr(snake, 's') and hasattr(snake, 'w') and \
                       hasattr(snake, 'h') and hasattr(snake, 'n2i') and \
                       hasattr(snake, '_compute_normals') and hasattr(snake, 'ndims'):
                        
                        axes[0].set_title("Input Image + Snake Overlay")
                        snake_disp_graph = snake.getGraph() # Gets graph with nodes on CPU
                        for u, v in snake_disp_graph.edges():
                            pos_u = snake_disp_graph.nodes[u]['pos']
                            pos_v = snake_disp_graph.nodes[v]['pos']
                            axes[0].plot([-pos_u[0], -pos_v[0]], [pos_u[1], pos_v[1]], 'magenta', linewidth=1.2)

                        s_cpu = snake.s.detach().cpu()
                        w_cpu = snake.w.detach().cpu() / 2.0 # half-widths
                        
                        current_normals_tuple = snake._compute_normals(snake.s) # snake.s is on device
                        normals_at_nodes_cpu = None
                        if snake.ndims == 2:
                            normals_at_nodes_cpu = current_normals_tuple[0].detach().cpu()
                        elif snake.ndims == 3: # For 3D, use the first normal component (n1)
                            normals_at_nodes_cpu = current_normals_tuple[0].detach().cpu()

                        if normals_at_nodes_cpu is not None:
                            node_to_idx = snake.n2i
                            for u_node_id, v_node_id in snake.h.edges(): # snake.h is internal graph
                                if u_node_id in node_to_idx and v_node_id in node_to_idx:
                                    u_idx = node_to_idx[u_node_id]
                                    v_idx = node_to_idx[v_node_id]

                                    pos_u_tensor = s_cpu[u_idx]
                                    pos_v_tensor = s_cpu[v_idx]
                                    width_u_val = w_cpu[u_idx]
                                    width_v_val = w_cpu[v_idx]
                                    
                                    original_normal_u_vec = normals_at_nodes_cpu[u_idx].numpy()
                                    original_normal_v_vec = normals_at_nodes_cpu[v_idx].numpy()

                                    # Rotate normal vector 90-deg CCW: (ny,nx) -> (-nx,ny)
                                    # original_normal_u_vec[0] is ny, original_normal_u_vec[1] is nx
                                    corrected_normal_u_vec = np.array([-original_normal_u_vec[1], original_normal_u_vec[0]])
                                    corrected_normal_v_vec = np.array([-original_normal_v_vec[1], original_normal_v_vec[0]])

                                    # Calculate width points with corrected normals
                                    # pos_u_tensor is (y,x). corrected_normal_u_vec is (new_ny, new_nx)
                                    p_Lu = (pos_u_tensor.numpy() - width_u_val.item() * corrected_normal_u_vec)
                                    p_Ru = (pos_u_tensor.numpy() + width_u_val.item() * corrected_normal_u_vec)
                                    p_Lv = (pos_v_tensor.numpy() - width_v_val.item() * corrected_normal_v_vec)
                                    p_Rv = (pos_v_tensor.numpy() + width_v_val.item() * corrected_normal_v_vec)
                                    
                                    # Plot using [1] for x, [0] for y after CCW rotation
                                    # p_Lu is (y,x). Rotated plot x is -p_Lu[0], y is p_Lu[1]
                                    axes[0].plot([-p_Lu[0], -p_Lv[0]], [p_Lu[1], p_Lv[1]], color='cyan', linewidth=1.5)
                                    axes[0].plot([-p_Ru[0], -p_Rv[0]], [p_Ru[1], p_Rv[1]], color='cyan', linewidth=1.5)
                    
                    # Ensure colorbar is associated with the image, not potentially overwritten by snake plots
                    if axes[0].images: # Check if an image was plotted
                        fig.colorbar(axes[0].images[0], ax=axes[0]) 
                    
                    fig.colorbar(axes[0].images[0], ax=axes[0]) # Add colorbar for the image
                    axes[0].axis('off')

                    # plot ground-truth label
                    im1 = axes[1].imshow(lbl_to_plot, cmap='viridis')
                    fig.colorbar(im1, ax=axes[1])
                    axes[1].set_title("Ground Truth Label")
                    axes[1].axis('off')

                    # plot network prediction
                    im2 = axes[2].imshow(prd_to_plot, cmap='viridis')
                    fig.colorbar(im2, ax=axes[2])
                    axes[2].set_title("Network Prediction")
                    axes[2].axis('off')

                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                    plot_filename = os.path.join("./trainplot/", f"epoch_{iterations}_batch_visualization_{idx}.png")
                    plt.savefig(plot_filename)
                    plt.close(fig)
                    logger.info(f"Saved visualization to {plot_filename}")
                except Exception as e:
                    logger.error(f"Error during plotting/saving visualization: {e}")
                
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
        local_output_path = "./val_plot/"
        drive_valid_path = os.path.join(drive_output_path, "output_valid")
        local_valid_path = os.path.join(local_output_path, "output_valid")
        plot_save_path = os.path.join(local_output_path, "plots")
        utils.mkdir(drive_valid_path)
        utils.mkdir(local_valid_path)
        utils.mkdir(plot_save_path)

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

                image_np = utils.from_torch(image)[0] # Get the first image in batch for plotting
                pred_np = utils.from_torch(pred)[0]
                preds.append(pred_np)
                label_np = utils.from_torch(label)[0]

                pred_np_for_skeleton = pred_np[0] if pred_np.ndim == 3 else pred_np
                label_np_for_skeleton = label_np[0] if label_np.ndim == 3 else label_np

                pred_mask_skeleton = skeletonize(pred_np_for_skeleton <= 5) # No division by 255 needed for boolean
                label_mask_skeleton = (label_np_for_skeleton == 0)

                corr, comp, qual = correctness_completeness_quality(pred_mask_skeleton, label_mask_skeleton, slack=3)
                
                scores["corr"].append(corr)
                scores["comp"].append(comp)
                scores["qual"].append(qual)
                
                # Save input and ground truth (only once)
                input_filename_local = os.path.join(local_valid_path, "val_input_{:03d}.npy".format(i))
                if not os.path.exists(input_filename_local):
                    np.save(input_filename_local, image_np)

                gt_filename_local = os.path.join(local_valid_path, "val_gt_{:03d}.npy".format(i))
                if not os.path.exists(gt_filename_local):
                    np.save(gt_filename_local, label_np)

                # Save predictions to BOTH Google Drive and local path
                # 1. Save to Google Drive
                drive_pred_filename = os.path.join(drive_valid_path, "val_pred_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(drive_pred_filename, pred_np)
                
                drive_pred_mask_filename = os.path.join(drive_valid_path, "val_predmask_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(drive_pred_mask_filename, pred_mask_skeleton.astype(np.uint8)) # Save skeleton as uint8
                
                # 2. Save to local path
                local_pred_filename = os.path.join(local_valid_path, "val_pred_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(local_pred_filename, pred_np)
                
                local_pred_mask_filename = os.path.join(local_valid_path, "val_predmask_{:03d}_epoch_{:06d}.npy".format(i, iteration))
                np.save(local_pred_mask_filename, pred_mask_skeleton.astype(np.uint8)) # Save skeleton as uint8

                # Plotting for the first few validation samples or periodically
                if i < 3 or iteration % 10 == 0: # Example: plot for first 3 samples or every 10 iterations
                    try:
                        # Prepare for plotting (select first channel if multiple, ensure 2D)
                        img_to_plot = image_np[0] if image_np.ndim == 3 else image_np
                        lbl_to_plot = label_np[0] if label_np.ndim == 3 else label_np
                        prd_to_plot = pred_np[0] if pred_np.ndim == 3 else pred_np

                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        fig.suptitle(f"Validation - Iteration {iteration} - Sample {i}")

                        # Plot Input Image
                        im0 = axes[0].imshow(img_to_plot, cmap='gray')
                        fig.colorbar(im0, ax=axes[0])
                        axes[0].set_title("Input Image")
                        axes[0].axis('off')

                        # Plot Ground Truth Label and its skeleton
                        im1 = axes[1].imshow(lbl_to_plot, cmap='viridis')
                        fig.colorbar(im1, ax=axes[1])
                        axes[1].imshow(label_mask_skeleton, cmap='Reds', alpha=0.5, interpolation='none') # Overlay skeleton
                        axes[1].set_title("Ground Truth + GT Skel (Red)")
                        axes[1].axis('off')

                        # Plot Prediction and its skeleton
                        im2 = axes[2].imshow(prd_to_plot, cmap='viridis')
                        fig.colorbar(im2, ax=axes[2])
                        axes[2].imshow(pred_mask_skeleton, cmap='Blues', alpha=0.5, interpolation='none') # Overlay skeleton
                        axes[2].set_title("Prediction + Pred Skel (Blue)")
                        axes[2].axis('off')
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.95])
                        plot_filename = os.path.join(plot_save_path, f"val_plot_iter_{iteration}_sample_{i}.png")
                        plt.savefig(plot_filename)
                        plt.close(fig)
                        logger.info(f"Saved validation plot to {plot_filename}")
                    except Exception as e:
                        logger.error(f"Error during validation plotting: {e}")

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
    
    
    
