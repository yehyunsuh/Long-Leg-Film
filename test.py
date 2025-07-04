import torch 
import math
import time
import os 
import csv
import numpy as np

from dataset import load_data
from tqdm import tqdm
from utility.log import log_terminal, log_test_results, log_test_results_with_angle
from utility.train import rmse, geom_element, angle_element, extract_pixel
from utility.visualization import visualize


def test(args, model, DEVICE):
    print("=====Starting Testing Process=====")
    os.makedirs(f'{args.result_directory}/{args.wandb_name}/test', exist_ok=True)
    os.makedirs(f'{args.result_directory}/{args.wandb_name}/test_angle', exist_ok=True)
    _, _, test_loader = load_data(args)

    model.eval()
    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    extracted_pixels_to_df = []
    rmse_list = [[0]*len(test_loader) for _ in range(args.output_channel)]
    angle_list = [[0]*len(test_loader) for _ in range(len(args.label_for_angle))]
    angle_total = []
    label_total = []

    extracted_pixels_list = []
    with torch.no_grad():
        start = time.time()
        # for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(test_loader)):
        #     image = image.to(DEVICE)
        #     label = label.to(DEVICE)
        #     image_path = image_path[0]
        #     image_name = image_name[0].split('.')[0]
        #     prediction = model(image)
        #     label_total.append(np.ndarray.tolist(np.array(torch.Tensor(label_list), dtype=object).reshape(args.output_channel*2,1)))

        #     # validate angle difference
        #     if args.label_for_angle != []:
        #         predict_angle, label_angle = angle_element(args, prediction, label_list, DEVICE)
        #         angle_total.append([image_name] + predict_angle + label_angle)
        #         for i in range(len(args.label_for_angle)):
        #             angle_list[i][idx] = abs(label_angle[i] - predict_angle[i])

        #     # validate mean geom difference
        #     predict_spatial_mean, label_spatial_mean = geom_element(torch.sigmoid(prediction), label)

        #     ## get rmse difference
        #     rmse_list, index_list = rmse(args, prediction, label_list, idx, rmse_list)
        #     extracted_pixels_list.append(index_list)

        #     extracted_pixels_array = np.array(index_list[0]).reshape(-1)
        #     tmp_list = [f'{image_name}.png', len(index_list[0])]
        #     for i in range(len(extracted_pixels_array)):
        #         tmp_list.append(extracted_pixels_array[i])
        #     extracted_pixels_to_df.append(tmp_list)

        #     ## make predictions to be 0. or 1.
        #     prediction_binary = (prediction > 0.5).float()
        #     dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum() + 1e-8)

        #     ## visualize
        #     # print(image_name, predict_angle)
        #     visualize(
        #         args, idx, image_path, image_name, label, label_list, None, extracted_pixels_list, prediction, prediction_binary,
        #         predict_spatial_mean, label_spatial_mean, predict_angle + label_angle, 'test'
        #     )

        # Case: No labels for test dataset
        for idx, (image, image_path, image_name) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            prediction = model(image)

            index_list = extract_pixel(args, prediction)
            extracted_pixels_list.append(index_list)

            extracted_pixels_array = np.array(index_list[0]).reshape(-1)
            tmp_list = [f'{image_name}.png', len(index_list[0])]
            for i in range(len(extracted_pixels_array)):
                tmp_list.append(extracted_pixels_array[i])
            extracted_pixels_to_df.append(tmp_list)

            # Case: No labels for test dataset
            if args.label_for_angle != []:
                predict_angle = angle_element(args, prediction, None, 'test', DEVICE)
                angle_total.append([image_name] + predict_angle)

            ## visualize
            visualize(
                args, idx, image_path, image_name, None, None, None, extracted_pixels_list, prediction, None,
                None, None, predict_angle, 'test'
            )
            
        end = time.time()

    print("=====Testing Process Done=====")
    print(f"{end - start:.5f} seconds for {len(test_loader)} images")

    # dice = dice_score/len(test_loader)
    # rmse_mean_by_label = []
    # for i in range(len(rmse_list)):
    #     tmp_sum, count = 0, 0
    #     for j in range(len(rmse_list[i])):
    #         if rmse_list[i][j] != -1:
    #            tmp_sum += rmse_list[i][j]
    #            count += 1
    #     rmse_mean_by_label.append(tmp_sum/count)

    # total_rmse_mean = sum(rmse_mean_by_label)/len(rmse_mean_by_label)
    if args.label_for_angle != []:
        # add up angle values
        angle_value = []
        for i in range(len(args.label_for_angle)):
            angle_value.append(sum(angle_list[i]))
        angle_value.append(sum(list(map(sum, angle_list))))

    # if args.wandb and args.label_for_angle == []:
    #     log_test_results(
    #         dice, total_rmse_mean, rmse_mean_by_label
    #     )
    # elif args.wandb and args.label_for_angle != []:
    #     log_test_results_with_angle(
    #         dice, total_rmse_mean, rmse_mean_by_label, 
    #         angle_value, len(test_loader), len(args.label_for_angle)
    #     )

    log_terminal(args, "test_prediction", extracted_pixels_list)
    # log_terminal(args, "test_label", label_total)
    # log_terminal(args, "test_rmse", rmse_list)
    log_terminal(args, "test_angle", angle_total)

    row_name = ["image_name", "number_of_labels"]
    for i in range(args.output_channel):
        row_name.append(f'label_{i}_y')
        row_name.append(f'label_{i}_x')
    csv_path = f'results/{args.wandb_name}/{args.wandb_name}_test_prediction.csv'
    with open(csv_path, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(row_name)
        write.writerows(sorted(extracted_pixels_to_df))

    # create csv from angle_total
    if args.label_for_angle != []:
        row_name = ["image_name"]
        row_name.append('LDFA')
        row_name.append('MPTA')
        row_name.append('MHKA')
        row_name.append('Long MHKA')
        csv_path = f'results/{args.wandb_name}/{args.wandb_name}_test_angle.csv'
        with open(csv_path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(row_name)
            write.writerows(sorted(angle_total))