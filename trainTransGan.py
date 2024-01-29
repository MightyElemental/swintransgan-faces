from util.arghandler import parse_args

args = parse_args()

import util.util as util
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from model.discriminator_conditional import Discriminator
import torchvision.utils as utils
import numpy as np
import os
import util.webhook as webhook
from tqdm import tqdm # progress bar
import time
if args.transformer_type == "full":
    from model.generator_transformer import Generator
elif args.transformer_type == "swin":
    from model.generator_transformer import SwinGenerator as Generator

# -= Settings =-
beta1disc = 0.5 # Beta1 hyperparam for Adam optimizers
beta1gen = 0.5
seed = time.time()

# -= Set up device =-
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

if args.verbose: print(f"PyTorch Version: {torch.__version__}\n")

if not args.disable_webhook: webhook.init_webook(args.webhook_file)



# -------------
# SETUP DATASET
# -------------

class_count = 0
if args.train:
    print("Loading dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(args.size, antialias=True), transforms.CenterCrop(args.size)])
    raw_data = datasets.CelebA(root="data/", split="train", target_type=["attr"], download=True, transform=transform)
    #raw_data = torch.utils.data.Subset(raw_data, range(batch_size*1000))
    dataloader = torch.utils.data.DataLoader(raw_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, prefetch_factor=4)
    class_names = raw_data.attr_names
    if args.conditional:
        class_count = len(raw_data.attr[0]) 
        print(f"Using {class_count} classes\n")



# --------------
# LOAD ARGUMENTS
# --------------

current_epoch = 0

#layer_map = {2**x:int(val) for x, val in enumerate(args.trans_layers.split(","), start=3)}
layer_map = [int(x) for x in args.trans_layers.split(",")]

nhead = [int(x) for x in args.nheads.split(",")]

loaded_checkpoint = False
if not args.do_not_resume:
    latest_cp = util.get_latest_checkpoint(args.path_cp) if args.ckpt_file is None else args.ckpt_file
    ckpt_path = f"{args.path_cp}{latest_cp}" if args.ckpt_file is None else args.ckpt_file
    if latest_cp:
        checkpoint = torch.load(ckpt_path, map_location=device)
        current_epoch = checkpoint["epoch"]+1
        if args.train:
            try: seed = checkpoint["seed"]
            except: print("no seed found")
        # generator
        gen_cp = checkpoint["generator"]
        # generator arch
        gen_arch = gen_cp["arch"]
        layer_map = gen_arch["layer_map"]
        args.latent_size = gen_arch["latent_size"]
        class_count = gen_arch["class_count"]
        args.size = gen_arch["size"]
        nhead = gen_arch["nheads"]
        args.channel_multiplier = gen_arch["channel_multiplier"]
        args.window_size = gen_arch["window_size"]
        # discriminator
        disc_cp = checkpoint["discriminator"]
        args.ndf = disc_cp["arch"]["ndf"]

        loaded_checkpoint = True
        print(f"Loading checkpoint file {latest_cp}\n")
    else:
        print("No existing checkpoint found. Starting new training.\n")

if args.verbose: print("Using seed",seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)



# ---------------
# DEFINE NETWORKS
# ---------------

gen_net = Generator(args.latent_size, img_size=args.size, class_count=class_count, layer_map=layer_map, channel_mul=args.channel_multiplier, nhead=nhead, window_size=args.window_size)
disc_net = Discriminator(args.ndf, image_size=args.size, class_count=class_count)
gen_net.to(device)
disc_net.to(device)

# Parallelize model across multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    gen_net = nn.DataParallel(gen_net)

# Setup Adam optimizers for both Generator and Discriminator
optimizer_discrim = optim.Adam(disc_net.parameters(), lr=args.learning_rate, betas=(beta1disc, 0.999))
optimizer_gen = optim.Adam(gen_net.parameters(), lr=args.learning_rate, betas=(beta1gen, 0.999))
#scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen,patience=5,factor=0.5,verbose=args.verbose)

# Load checkpoints
if loaded_checkpoint:
    gen_net.load_state_dict(gen_cp["model_state"])
    optimizer_gen.load_state_dict(gen_cp["optimizer_state"])
    disc_net.load_state_dict(disc_cp["model_state"])
    optimizer_discrim.load_state_dict(disc_cp["optimizer_state"])
    print("Loaded states")

print()
print(f"Generator has {util.count_parameters(gen_net):,} parameters")
print(f"Discriminator has {util.count_parameters(disc_net):,} parameters")
print()
util.print_mem_usage()
print()

# -= Define =-
criterion = nn.BCELoss().to(device)

real_label = 1.0
fake_label = 0.0

# Initialize loss history arrays
g_loss_hist = []
d_loss_hist = []
g_epoch_loss_hist = []
d_epoch_loss_hist = []

if loaded_checkpoint:
    g_loss_hist = gen_cp["loss_history"]
    g_epoch_loss_hist = gen_cp["epoch_loss_history"]
    d_loss_hist = disc_cp["loss_history"]
    d_epoch_loss_hist = disc_cp["epoch_loss_history"]
    print("Loaded loss histories\n")

if args.print_settings:
    print(args)
    print()

# Generate folders if needed
util.make_path_if_not_exist(args.path_cp)
util.make_path_if_not_exist(args.path_img_cp)



# ========== NO MORE CHECKPOINT LOADING PAST THIS POINT ==========



# ------------------------
# DEFINE CHECKPOINT SAVING
# ------------------------

# -= Define checkpoint image vectors =-
z_cp = torch.randn(args.img_count_cp, args.latent_size, device=device)
c_cp = torch.randint(2, (args.img_count_cp, class_count), device=device)

def save_img_checkpoint(path:str, z_cp:torch.Tensor, c_cp:torch.Tensor, img_count_cp:int, epoch:int, batch:int):
    """Save a batch of images using existing latent vector

    Args:
        path (str): the folder path to save the image
        z_cp (torch.Tensor): the latent vector
        c_cp (torch.Tensor): the class encoding vector
        img_count_cp (int): the number of images to generate
        epoch (int): the current epoch
        batch (int): the current batch

    Returns:
        str: the path to the img
    """
    util.make_path_if_not_exist(path)
    with torch.no_grad():
        imgs=gen_net(z_cp, c_cp)
        #imgs = transforms.Normalize(0, 5)(imgs) # TODO: Remove
    img_path = f"{path}e{epoch:04d}-b{batch:04d}.jpg"
    utils.save_image(imgs, img_path, nrow=int(img_count_cp**0.5))
    return img_path

def save_checkpoint(epoch:int):
    torch.save({
        "epoch": epoch,
        "seed": seed,
        "generator": {
            "model_state": gen_net.state_dict(),
            "optimizer_state": optimizer_gen.state_dict(),
            #"scheduler_state": scheduler_gen.state_dict(),
            "loss_history": g_loss_hist,
            "epoch_loss_history": g_epoch_loss_hist,
            "arch": {
                "size": args.size,
                "latent_size": args.latent_size,
                "layer_map": layer_map,
                "class_count": class_count,
                "channel_multiplier": args.channel_multiplier,
                "nheads": nhead,
                "window_size": args.window_size
            },
        },
        "discriminator": {
            "model_state": disc_net.state_dict(),
            "optimizer_state": optimizer_discrim.state_dict(),
            #"scheduler_state": scheduler_disc.state_dict(),
            "loss_history": d_loss_hist,
            "epoch_loss_history": d_epoch_loss_hist,
            "arch": {
                "ndf": args.ndf,
            },
        },
    }, f"{args.path_cp}checkpoint-{epoch:04d}.pt")




# --------------------
# DEFINE TRAINING LOOP
# --------------------

def train(epoch:int, max_epoch:int):
    tqRange = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch:04d}]",
            position=1,
            leave=False,
        )
    gen_losses = []
    disc_losses = []
    for batch, (images, labels) in tqRange:
        optimizer_discrim.zero_grad()
        optimizer_gen.zero_grad()

        # --== Train Discriminator ==--
        # -= Train with batch of real images =-
        output = disc_net(images.to(device), labels.to(device)).view(-1)
        label = torch.full((len(output),), real_label, dtype=torch.float, device=device)
        d_loss_real = criterion(output, label)
        d_loss_real.backward()
        #D_x = output.mean().item()

        # -= Train with batch of fake images =-
        # Create a batch of random latent vectors the same length as the current training batch
        z = torch.randn(len(label), args.latent_size, device=device)
        c = torch.randint(2, (len(label), class_count), device=device)
        # Train on fake data
        fake = gen_net(z, c)
        label.fill_(fake_label)
        output = disc_net(fake.detach(), labels.to(device)).view(-1)
        d_loss_fake = criterion(output, label)
        d_loss_fake.backward()
        d_loss = d_loss_real + d_loss_fake
        disc_losses.append(d_loss.item())

        optimizer_discrim.step()

        # --== Train Generator ==--
        label.fill_(real_label)  # labels are inverted for Generator training
        output = disc_net(fake, c).view(-1)
        g_loss = criterion(output, label)
        gen_losses.append(g_loss.item())
        g_loss.backward()

        optimizer_gen.step()

        if batch % 50 == 0:
            g_loss_hist.append(g_loss.item())
            d_loss_hist.append(d_loss.item())
            util.save_graph(f"{args.path_cp}loss.jpg", "Training Loss", "Batch x50", [g_loss_hist, d_loss_hist], ["Generator", "Discriminator"])
            alloc_gpu_mem = sum(util.get_gpu_alloc_mem())
            cached_gpu_mem = sum(util.get_gpu_cached_mem())
            tqRange.set_postfix_str(f"{alloc_gpu_mem/(1024**2):.0f}+{cached_gpu_mem/(1024**2):.0f} MiB")
        if batch % args.batch_checkpoint == 0:
            img_path = save_img_checkpoint(args.path_img_cp, z_cp, c_cp, args.img_count_cp, epoch, batch)
            alloc_gpu_mem = sum(util.get_gpu_alloc_mem())
            cached_gpu_mem = sum(util.get_gpu_cached_mem())
            total_gpu_mem = alloc_gpu_mem + cached_gpu_mem

            if not args.disable_webhook: 
                webhook.submit("GAN Training Update",
                            f"Epoch {epoch:04d} / Batch {batch:04d}\n"\
                            f"g_loss {g_loss.item():.4f} / d_loss {d_loss.item():.4f}\n\n"\
                            f"Allocated GPU Mem: {alloc_gpu_mem/(1024**2):.1f} MiB\n"\
                            f"Cached GPU Mem: {cached_gpu_mem/(1024**2):.1f} MiB\n"\
                            f"Total GPU Memory: {total_gpu_mem/(1024**2):.1f} MiB",
                            "TransGAN", img_path)
            # Detect total collapse
            if args.detect_collapse and len(d_loss_hist) > 15:
                if np.sum(d_loss_hist[-15:]) < 0.05 and np.sum(g_loss_hist[-15:]) < 0.05:
                    if not args.disable_webhook: webhook.submit("GAN Training COLLAPSED", f"Epoch {epoch:04d} / Batch {batch:04d}", "Trans-Conv")
                    raise util.CollapseError("Total Collapse")

    d_epoch_loss_hist.append( np.mean(disc_losses) )
    g_epoch_loss_hist.append( np.mean(gen_losses) )
    #scheduler_gen.step( np.mean(gen_losses) )
    util.save_graph(f"{args.path_cp}loss-epoch.jpg", "Training Loss (avg)", "Epoch", [g_epoch_loss_hist, d_epoch_loss_hist], ["Generator", "Discriminator"])




# ========== NO MORE FUNCTION DEFINITIONS PAST THIS POINT ==========


# --------------
# START TRAINING
# --------------

if args.train: # If in training mode
    torch.autograd.set_detect_anomaly(True)

    max_epoch = args.epochs
    #if max_epoch > 0:
    #    max_epoch += current_epoch

    if current_epoch > 0:
        print(f"Continuing training from checkpoint epoch {current_epoch}")

    if not args.disable_webhook: webhook.submit("GAN Training Started", f"Gen Params: {util.count_parameters(gen_net):,}\nDisc Params: {util.count_parameters(disc_net):,}\n\n{args}", "Trans-Conv")

    count = 0

    t1 = time.time()
    with tqdm(total=max_epoch,desc="Training Epoch", initial=current_epoch, leave=False, position=0) as pbar:
        while max_epoch < 0 or current_epoch < max_epoch:
            pbar.set_description_str(f"Training Epoch [{current_epoch:04d}/{max_epoch if max_epoch > 0 else 'inf'}]")
            train(current_epoch, max_epoch)

            save_checkpoint(current_epoch)
            if not args.disable_webhook:
                webhook.submit("GAN Training Loss", f"Epoch {current_epoch:04d}\nTotal Runtime: {time.time()-t1:,.0f}s", "Trans-Conv", f"{args.path_cp}loss.jpg")
                webhook.submit("GAN Training Loss (Epoch)", f"Epoch {current_epoch:04d}\n g_loss {g_epoch_loss_hist[-1]:.4f} / d_loss {d_epoch_loss_hist[-1]:.4f}", "TransGAN", f"{args.path_cp}loss-epoch.jpg")
            current_epoch += 1
            count += 1
            pbar.update(1)
        pbar.close()
    run_time = time.time()-t1

    cmptl_msg = f"Ran through {count} epochs in {run_time:,.0f} seconds"
    if not args.disable_webhook: webhook.submit("GAN Training Completed", cmptl_msg, "Trans-Conv")
    print("Training complete.",cmptl_msg)

else: # If in evaluation mode
    # randomize seed
    torch.cuda.manual_seed_all(time.time())

    # import eval tools
    from util.inception import inception_score
    from cleanfid import fid

    with torch.no_grad():

        GEN_BATCH = args.batch_size
        assert args.evaluate % GEN_BATCH == 0, "Evaluation image count must be a multiple of the batch size!"
        
        path = args.eval_path
        datapath = f"{path}data/"

        if not os.path.isdir(datapath):
            os.makedirs(datapath)
        for i in tqdm(range(0,args.evaluate,GEN_BATCH), desc="Generating Eval Imgs"):
            z = torch.randn(GEN_BATCH, args.latent_size, device=device)
            c = torch.randint(2, (GEN_BATCH, class_count), device=device)
            imgs = gen_net(z,c)
            for j, img in enumerate(imgs):
                img_path = f"{datapath}eval{i+j}.png"
                utils.save_image(img, img_path)

        # Run Eval here
        # FID
        print("Calculating FID")
        try:
            # Calculate dataset statistics
            if not fid.test_stats_exists("celeba", "clean"):
                print("CelebA stats does not exist. Calculating and caching statistics...")
                fid.make_custom_stats("celeba", args.eval_db, mode="clean", num_workers=args.workers)

            fid_score = fid.compute_fid(datapath, dataset_name="celeba", mode="clean", dataset_split="custom")
        except ValueError as e:
            fid_score = "unable to calculate"
            print("Failed to generate FID score:",e)
        
        # IS
        if args.skip_is:
            is_mean, is_std = "Skipped", "Skipped"
        else:
            print("Calculating Inception Score")
            data = datasets.ImageFolder(root=path, transform=transforms.ToTensor())
            is_mean, is_std = inception_score(data, device, resize=True, batch_size=GEN_BATCH, num_workers=args.workers)

        result_text = ("==== Inception Score ====\n"
                      f"Mean = {is_mean} | std = {is_std}\n\n"
                       "==== Frechet Inception Distance ====\n"
                      f"FID = {fid_score}\n")

        print()
        print(result_text)

        util.text_to_file(result_text, f"{path}results.txt")
        




    print("Completed.")