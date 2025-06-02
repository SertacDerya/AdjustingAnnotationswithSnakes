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

def plot_ribbon_snake(
        ax,
        snake,
        crop_slices=None,
        face_color="cyan",
        face_alpha=0.35,
        edge_color="magenta",
        node_color="blue",
        node_size=10,
):
    if snake is None or snake.s is None or snake.s.numel() == 0:
        return

    ctrs = snake.s.detach().cpu().float().numpy()
    widths = snake.w.detach().cpu().float().numpy().ravel()

    if snake.ndims != 2:
        # logger.warning("plot_ribbon_snake currently only supports 2D snakes for full ribbon. Plotting nodes only.")
        # Optionally plot 3D nodes projected to 2D plane if desired (e.g. ctrs[:,1], ctrs[:,0] for x,y)
        if ctrs.shape[1] >=2: # Check if at least 2D coordinates exist
             ax.scatter(ctrs[:, 1], ctrs[:, 0], c=node_color, s=node_size, zorder=5, alpha=0.9, marker='x')
        return
    
    K = ctrs.shape[0]
    if K == 0 : return
    if K == 1: # Plot single node as a circle with its width
        # Calculate radius from width for plotting
        radius_px = widths[0] / 2.0 
        # For scatter, 's' is proportional to area, so (pi*r^2). Or simply scale node_size by width.
        # Here, let's just make size proportional to width for simplicity.
        ax.scatter(ctrs[0, 1], ctrs[0, 0], c=node_color, s=node_size * (widths[0] if widths[0]>0 else 1), zorder=5, alpha=0.9)
        return
    
    G = snake.getGraph()
    if G is None: return

    for u_name, v_name in G.edges():
        if u_name not in snake.n2i or v_name not in snake.n2i:
            logger.debug(f"Node {u_name} or {v_name} not in snake.n2i during plotting. Skipping edge.")
            continue
        iu, iv = snake.n2i[u_name], snake.n2i[v_name]

        if not (iu < K and iv < K): continue

        cu, cv = ctrs[iu], ctrs[iv]
        wu, wv = widths[iu], widths[iv]

        if not (np.isfinite(wu) and np.isfinite(wv) and wu > 0 and wv > 0):
            continue

        edge_vec = cv - cu
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-8: continue
        
        edge_tangent = edge_vec / edge_len
        edge_normal = np.array([-edge_tangent[1], edge_tangent[0]])

        # Define quad vertices in (y,x)
        # Order: start_left, end_left, end_right, start_right (A, B, C, D)
        quad_coords_yx = np.array([
            cu - edge_normal * wu * 0.5,  # A
            cv - edge_normal * wv * 0.5,  # B
            cv + edge_normal * wv * 0.5,  # C
            cu + edge_normal * wu * 0.5   # D
        ])
        
        if not np.all(np.isfinite(quad_coords_yx)): continue
            
        # Convert to (x,y) for plt.Polygon
        quad_coords_xy = quad_coords_yx[:, ::-1] 

        patch = plt.Polygon(quad_coords_xy, closed=True,
                            facecolor=face_color, alpha=face_alpha,
                            edgecolor=None) # Use None or a specific color for edge
        ax.add_patch(patch)

        ax.plot([cu[1], cv[1]], [cu[0], cv[0]],
                color=edge_color, linewidth=0.8, alpha=0.7)

    ax.scatter(ctrs[:, 1], ctrs[:, 0],
               c=node_color, s=node_size, zorder=5, alpha=0.9)


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
    def __init__(self, dataloader, ours=False, ours_start=0): # ours_loss_config can be a dict
        self.dataloader = dataloader
        self.ours = ours
        self.ours_start = ours_start
        self.plot_dir = "./trainplot/"
        utils.mkdir(self.plot_dir)

    def __call__(self, iterations, network, optimizer, lr_scheduler, base_loss, our_loss):

        mean_loss = 0.0
        num_valid_batches = 0
        network.train() 
        for idx, (images, labels, masks, graphs, slices) in enumerate(self.dataloader):

            images = images.cuda()
            labels = labels.cuda()
            
            preds = network(images.contiguous())
            snake_for_plotting = None

            if self.ours and iterations >= self.ours_start:
                loss, snake_for_plotting, dmap = our_loss(preds, graphs, slices, iterations)
            else:
                loss = base_loss(preds, labels)
            
            loss_v = loss.item()

            # Plotting (first batch of every 10th iteration)
            if iterations % 10 == 0:
                try:
                    img_np = utils.from_torch(images[0].cpu()) 
                
                    if img_np.max() > 1.0 and img_np.min() >=0: img_np = img_np / 255.0
                    
                    lbl_np = utils.from_torch(labels[0].cpu())
                    prd_np = utils.from_torch(preds[0].cpu())

                    dmap_np = utils.from_torch(dmap[0].cpu())
                
                    if img_np.shape[0] == 3: img_np = img_np.transpose(1,2,0)
                    elif img_np.shape[0] == 1: img_np = img_np.squeeze(0)

                    lbl_np = lbl_np.squeeze(0) if lbl_np.ndim == 3 and lbl_np.shape[0]==1 else lbl_np
                
                    prd_np = prd_np.squeeze(0) if prd_np.ndim == 3 and prd_np.shape[0]==1 else prd_np

                    fig, axes = plt.subplots(2, 2, figsize=(12,12))
                    fig.suptitle(f"Train - Epoch {iterations} - Batch {idx} (Loss: {loss_v:.4f})")

                    axes[0][0].imshow(img_np, cmap="gray" if img_np.ndim==2 else None)
                    axes[0][0].set_title("Input + Snake")
                    if snake_for_plotting:
                        plot_ribbon_snake(axes[0][0], snake_for_plotting, 
                                          crop_slices=slices[0] if slices and len(slices) > 0 else None)
                    axes[0][0].axis("off")

                    im1 = axes[0][1].imshow(lbl_np, cmap="viridis")
                    axes[0][1].set_title("GT Label + Snake")
                    fig.colorbar(im1, ax=axes[0][1])
                    if snake_for_plotting:
                         plot_ribbon_snake(axes[0][1], snake_for_plotting,
                                          crop_slices=slices[0] if slices and len(slices) > 0 else None)
                    axes[0][1].axis("off")

                    im2 = axes[1][0].imshow(prd_np, cmap='viridis')
                    axes[1][0].set_title("Prediction")
                    fig.colorbar(im2, ax=axes[1][0])
                    axes[1][0].axis('off')

                    im3 = axes[1][1].imshow(dmap_np, cmap='viridis')
                    axes[1][1].set_title("Distance Map")
                    fig.colorbar(im3, ax=axes[1][1])
                    axes[1][1].axis('off')

                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.savefig(os.path.join(self.plot_dir, f"epoch_{iterations}_batch_{idx}_train_vis.png"))
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Plotting error iter {iterations}, batch {idx}: {e}", exc_info=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss += loss_v
            num_valid_batches +=1


            if lr_scheduler is not None:
                lr_scheduler.step() 

        avg_loss = mean_loss / num_valid_batches if num_valid_batches > 0 else float('nan')
        return {"loss": float(avg_loss)}
    
    
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

                pred_mask_skeleton = skeletonize(pred_np_for_skeleton <= 0) # No division by 255 needed for boolean
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
    
    
    
