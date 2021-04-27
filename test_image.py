import pandas
import os
import argparse
import time

from Inference import Inferencer


def main():
    parser = argparse.ArgumentParser()

    # Environment argument
    parser.add_argument('--mode', choices=['inference'], help='run mode')
    parser.add_argument('-cuda', action='store_true', help='Using GPU processor')
    parser.add_argument('-pin_memory', action='store_true', help='Load dataset while learning')
    parser.add_argument('--worker', type=int, default=1)

    # Inference parameter
    parser.add_argument('-save_figures', action='store_true', help='save figures')
    parser.add_argument('--points', type=int, default=64, help='The number of curve points')
    parser.add_argument('-clip_threshold', action='store_true', help='clip the output image to threshold of input')
    parser.add_argument('--model_path', type=str,
                        default='./model_checkpoints/CUD50.pt')

    # Data parameter
    parser.add_argument('--data_class', choices=['A', 'B'], help='input data class')
    parser.add_argument('--test_path', type=str,
                        default='',
                        help='Your awesome test image path')

    args = parser.parse_args()

    data_dir = os.path.join(args.test_path, args.data_class)
    saved_path = './outputs'

    data_file_dir = os.listdir(data_dir)

    inferencer = Inferencer(args)

    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    for i, fn in enumerate(data_file_dir):
        image_dir = data_dir + os.sep + fn

        output = inferencer.start_inference(image_dir)

        output.save(saved_path + '/' + fn, transparency=None, quality=100)
        
        print(fn + ' Done!!!')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time : %f' % (time.time() - start_time))
