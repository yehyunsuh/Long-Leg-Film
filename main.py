import argparse
import torch
import os

from utility.log import initiate_wandb
from model import UNet
from utility.preprocess import relocate, create_csv, pad_dataset
from test import test
from train import train
from train_until import train_until
from utility.main import arg_as_list, customize_seed, png_to_txt


def main(args):
    customize_seed(args.seed)
    png_to_txt()
    initiate_wandb(args)

    if args.preprocess:
        ## TODO: relocate images based on txt files
        relocate(args)

        ## TODO: dicom files into one folder
        create_csv(args)

        ## TODO: pad dicom files + change txt to csv
        ## TODO: resize values in csv that fits to 512
        pad_dataset(args)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')
    model = UNet(args,  DEVICE)

    # train model
    if args.train:
        if args.train_until:
            checkpoint= train_until(args, model, DEVICE)
        else:
            checkpoint = train(args, model, DEVICE)
    
    if args.test:
        ## Test Model
        if os.path.exists(f'./results/{args.wandb_name}/best.pth'):
            model.load_state_dict(torch.load(f'./results/{args.wandb_name}/best.pth')['state_dict'])
            
        else:
            model.load_state_dict(checkpoint['state_dict'])
        test(args, model, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## boolean arguments
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--no_visualization', action='store_true', help='whether to save image or not')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--geom_loss', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--no_reweight', action='store_true')
    parser.add_argument('--train_until', action='store_true')

    ## get dataset
    parser.add_argument('--excel_path', type=str, default="./xlsx/dataset.xlsx", help='path to dataset excel file')

    ## data preprocessing
    parser.add_argument('--dicom_path', type=str, default="./data/dicom", help='path to dicom dataset')
    parser.add_argument('--csv_path', type=str, default="./data/xlsx", help='path to csv dataset')
    parser.add_argument('--image_path_all', type=str, default="./data/image/all", help='where all the images are stored')
    parser.add_argument('--image_path', type=str, default="./data/image", help='path to image dataset')
    parser.add_argument('--image_padded_path', type=str, default="./data/image_padded", help='path to padded image')

    ## hyperparameters - data
    parser.add_argument('--dataset_csv', type=str, default="./data/xlsx/dataset.csv", help='train csv file path')
    parser.add_argument('--train_csv', type=str, default="./data/xlsx/train_dataset.csv", help='train csv file path')
    parser.add_argument('--test_csv', type=str, default="./data/xlsx/test_dataset.csv", help='test csv file path')
    parser.add_argument('--train_csv_preprocessed', type=str, default="./data/xlsx/train_dataset_preprocessed.csv", help='train csv file path')
    parser.add_argument('--test_csv_preprocessed', type=str, default="./data/xlsx/test_dataset_preprocessed.csv", help='test csv file path')
    parser.add_argument('--train_label_txt', type=str, default="./data/txt/train_label.txt", help='train label text file path')
    parser.add_argument('--test_label_txt', type=str, default="./data/txt/test_label.txt", help='test label text file path')
    parser.add_argument('--dataset_split', type=int, default=8, help='dataset split ratio')
    parser.add_argument('--dilate', type=int, default=65, help='dilate iteration')
    parser.add_argument('--dilation_decrease', type=int, default=10, help='dilation decrease in progressive erosion')
    parser.add_argument('--dilation_epoch', type=int, default=50, help='dilation per epoch')
    parser.add_argument('--connectivity', type=int, default=1)
    parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
    parser.add_argument('--batch_size', type=int, default=18, help='batch size')
    parser.add_argument('--train_threshold', type=float, default=0.8)
    
    ## hyperparameters - model
    parser.add_argument('--seed', type=int, default=2022, help='seed customization for result reproduction')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel size for UNet')
    parser.add_argument('--output_channel', type=int, default=11, help='output channel size for UNet')
    parser.add_argument('--encoder_depth', type=int, default=5, help='model depth for UNet')
    parser.add_argument("--decoder_channel", type=arg_as_list, default=[256,128,64,32,16], help='model decoder channels')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs')
    parser.add_argument('--angle_loss_weight', type=float, default=0)
    parser.add_argument("--label_for_angle", type=arg_as_list, default=[])
    

    ## hyperparameters - results
    parser.add_argument('--result_directory', type=str, default="./results", help='test label text file path')
    parser.add_argument('--threshold', type=float, default=0.5)

    ## wandb
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
    parser.add_argument('--wandb_sweep', action='store_true')
    parser.add_argument('--wandb_project', type=str, default="Long Leg Film", help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default="yehyun-suh", help='wandb entity name')
    parser.add_argument('--wandb_name', type=str, default="baseline", help='wandb name')

    args = parser.parse_args()
    main(args)