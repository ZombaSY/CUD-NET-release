from train import CUD
from Inference import Inferencer
from torch.cuda import is_available
from datetime import datetime

import argparse
import wandb
import os


def main():

    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    parser = argparse.ArgumentParser()

    # Environment argument
    parser.add_argument('--mode', choices=['train', 'calibrate', 'inference'], help='run mode')
    parser.add_argument('-cuda', action='store_true', help='Using GPU processor')
    parser.add_argument('--log_interval', type=int, default=5, help='Log interval per batch')
    parser.add_argument('-pin_memory', action='store_true', help='Load dataset while learning')
    parser.add_argument('--save_interval', type=int, default=100, help='save model interval to epoch')
    parser.add_argument('-use_wandb', action='store_true', help='utilize wandb')

    # Train parameter
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--worker', type=int, default=4)    # prevent overheading on using DEU filter.
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--points', type=int, default=64, help='The number of curve points')
    parser.add_argument('--model', type=str, help='Choose model')

    # Inference parameter
    parser.add_argument('-save_figures', action='store_true', help='save figures')
    parser.add_argument('-clip_threshold', action='store_true', help='clip the output image to threshold of input')
    parser.add_argument('--model_path', type=str,
                        default='./model_checkpoints/CUD500.pt')
    parser.add_argument('--data_path_x', type=str,
                        default='./images/T5-58_a.jpg')
    parser.add_argument('--data_path_y', type=str,
                        default='./images/T5-243_b.jpg')

    # Data parameter
    parser.add_argument('--saved_model_directory', type=str, default='model_checkpoints')
    parser.add_argument('--train_x_path', type=str,
                        default='',
                        help='Your awesome input image path')
    parser.add_argument('--train_y_path', type=str,
                        default='',
                        help='Your awesome target image path')
    parser.add_argument('--model_g_path', type=str,
                        default='model_checkpoints/CUD830.pt')

    args = parser.parse_args()
    now = datetime.now().strftime("%Y-%m-%d %H%M%S")

    if args.use_wandb:
        wandb.init(project='my project', config=args, name=now,
                   notes='train dataset = ' + args.train_x_path)
        print('Please check description in \'train.py\' loss function')

    print('Use CUDA :', args.cuda and is_available())

    if args.mode in ('train', 'calibrate'):
        cud = CUD(args, now)
        cud.start_train()

    elif args.mode in 'inference':
        inferencer = Inferencer(args)
        output = inferencer.start_inference(args.data_path_x, args.data_path_y)

        temp, _ = os.path.splitext(args.data_path_x)
        _, fn = os.path.split(temp)
        output.save('outputs/' + fn + '_out.jpg', transparency=None, quality=100)

    else:
        print('No mode supported.')


if __name__ == "__main__":
    main()
