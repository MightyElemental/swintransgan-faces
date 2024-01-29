import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Train Settings
    parser.add_argument("-s", "--size", type=int, default=32, help="Side length of the image. Must be a power of 2.")
    parser.add_argument("--ndf", type=int, default=64, help="The discriminator feature multiplier")
    parser.add_argument("-z", "--latent_size", type=int, default=100, help="The size of the latent vector")
    parser.add_argument("-D", "--do_not_resume", action="store_true", help="Should not resume training from latest checkpoint. No effect if checkpoint does not exist.")
    parser.add_argument("-C", "--conditional", action="store_true", help="Should train the model to be conditional")
    parser.add_argument("--path_cp", type=str, default="checkpoints/", help="The path for the checkpoint folder")
    parser.add_argument("--path_img_cp", type=str, default="checkpoints/preview/", help="The path for the image checkpoint folder")
    parser.add_argument("--trans_layers", type=str, default="5,2,2", help="The number of transformers in each layer in ascending order")
    parser.add_argument("--img_count_cp", type=int, default=25, help="The number of images to generate to show progress")
    parser.add_argument("-p", "--print_settings", action="store_true", help="Should the program print out the training settings")
    parser.add_argument("-e", "--epochs", type=int, default=-1, help="The maximum number of epochs to run the model. If a checkpoint is used, this number is added to the checkpoint epoch.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--batch_checkpoint", type=int, default=250, help="How many batches between checkpoint image generation")
    parser.add_argument("-m", "--channel_multiplier", type=int, default=2, help="The multiplier for the number of transformer channels")
    parser.add_argument("--nheads", type=str, default="4,4,4,4", help="The number of heads in the transformers")
    parser.add_argument("-W", "--disable_webhook", action="store_true")
    parser.add_argument("-T", "--train", action="store_true", help="Enable training mode")
    parser.add_argument("--window_size", default=4, type=int, help="The size of the Swin Transformer windows")
    parser.add_argument("-M","--detect_collapse", action="store_true", help="Enable detection of mode collapse and stop training when detected")
    parser.add_argument("--webhook_file", type=str, default="webhook.txt", help="The location of the file that contains the webhook url")
    parser.add_argument("-l","--learning_rate", type=float, default=0.0001)

    # Train & Eval Settings
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="The number of images in each batch")
    parser.add_argument("--transformer_type", "--trans_type", default="swin", type=str, choices=["full", "swin"], help="The type of transformer generator to use")
    parser.add_argument("--ckpt_file", type=str, default=None, help="The location of the checkpoint file to use")
    parser.add_argument("-j", "--workers", type=int, default=3, help="The number of worker threads to load data")

    # Evaluation Settings
    parser.add_argument("--evaluate", "--eval", default=12500, type=int, help="The number of images to generate in evaluation mode")
    parser.add_argument("--eval_path", type=str, default="evaluate/", help="The location to save evaluation image files")
    parser.add_argument("--eval_db", type=str, default="data/celeba/img_align_celeba/", help="The location of the dataset real images")
    parser.add_argument("--skip_is", action="store_true", help="Skip the Inception Score calculation")
    return parser.parse_args()