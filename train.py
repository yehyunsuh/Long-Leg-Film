import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pprint

from tqdm import tqdm
from utility.log import log_results, log_results_with_angle, log_terminal
from utility.train import set_parameters, rmse, geom_element, angle_element
from utility.visualization import visualize

def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_angle, optimizer, train_loader):
    total_loss, total_pixel_loss, total_angle_loss = 0, 0, 0
    model.train()
    for image, label, _, _, label_list in tqdm(train_loader):
        image = image.to(device=DEVICE)
        label = label.float().to(device=DEVICE)
        prediction = model(image)

        pixel_loss = loss_fn_pixel(prediction, label)
        if args.angle_loss_weight != 0 and args.label_for_angle != []:
            predict_angle, label_angle = angle_element(args, prediction, label_list, 'train', DEVICE)
            predict_angle = torch.Tensor(predict_angle).to(device=DEVICE)
            label_angle = torch.Tensor(label_angle).to(device=DEVICE)
            angle_loss = loss_fn_angle(predict_angle, label_angle)
            loss = pixel_loss + angle_loss * args.angle_loss_weight
        else:
            loss = loss_fn_pixel(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pixel_loss += pixel_loss.item()
        if args.angle_loss_weight != 0 and args.label_for_angle != []:
            total_angle_loss += angle_loss.item()

    return total_loss, total_pixel_loss, total_angle_loss


def validate_function(args, DEVICE, model, epoch, val_loader):
    print("=====Starting Validation=====")
    model.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    rmse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]
    angle_list = [[0]*len(val_loader) for _ in range(len(args.label_for_angle))]
    angle_total = []

    with torch.no_grad():
        for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(val_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            
            prediction = model(image)
            
            # validate angle difference
            if args.label_for_angle != []:
                predict_angle, label_angle = angle_element(args, prediction, label_list, 'train', DEVICE)
                angle_total.append(predict_angle + label_angle)
                for i in range(len(args.label_for_angle)):
                    angle_list[i][idx] = abs(label_angle[i] - predict_angle[i])

            # validate mean geom difference
            predict_spatial_mean, label_spatial_mean = geom_element(torch.sigmoid(prediction), label)

            ## get rmse difference
            rmse_list, index_list = rmse(args, prediction, label_list, idx, rmse_list)
            extracted_pixels_list.append(index_list)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum() + 1e-8)

            ## visualize
            if epoch % args.dilation_epoch == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1) or epoch % 50 == 0:
                if not args.no_visualization:
                    visualize(
                        args, idx, image_path, image_name, label, label_list, epoch, extracted_pixels_list, prediction, prediction_binary,
                        predict_spatial_mean, label_spatial_mean, predict_angle + label_angle, 'train'
                    )
    dice = dice_score/len(val_loader)

    # Removing RMSE for annotation that does not exist in the label
    rmse_mean_by_label = []
    for i in range(len(rmse_list)):
        tmp_sum, count = 0, 0
        for j in range(len(rmse_list[i])):
            if rmse_list[i][j] != -1:
               tmp_sum += rmse_list[i][j]
               count += 1
        rmse_mean_by_label.append(tmp_sum/count)

    total_rmse_mean = sum(rmse_mean_by_label)/len(rmse_mean_by_label)
    print(f"Dice score: {dice}")
    print(f"Average Pixel to Pixel Distance: {total_rmse_mean}")

    if args.label_for_angle != []:
        # add up angle values
        angle_value = []
        for i in range(len(args.label_for_angle)):
            angle_value.append(sum(angle_list[i]))
        angle_value.append(sum(list(map(sum, angle_list))))

        return dice, total_rmse_mean, rmse_list, rmse_mean_by_label, angle_total, angle_value
    else:
        return dice, total_rmse_mean, rmse_list, rmse_mean_by_label, 0, 0


def train(args, model, DEVICE):
    best_loss, best_rmse_mean, best_angle_diff = np.inf, np.inf, np.inf
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")
        
        if epoch % args.dilation_epoch == 0:
            args, loss_fn_pixel, loss_fn_angle, train_loader, val_loader = set_parameters(
                args, model, epoch, DEVICE
            )

        loss, pixel_loss, angle_loss = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_angle, optimizer, train_loader
        )
        dice, rmse_mean, rmse_list, rmse_mean_by_label, angle_list, angle_value = validate_function(
            args, DEVICE, model, epoch, val_loader
        )

        print("Average Train Loss: ", loss/len(train_loader))
        if best_loss > loss:
            print("=====New best model=====")
            best_loss = loss

        if best_rmse_mean > rmse_mean:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }
            # torch.save(checkpoint, f'./results/{args.wandb_name}/best.pth')
            best_rmse_mean = rmse_mean
            best_rmse_list = rmse_list
        
        if args.label_for_angle != []:
            if best_angle_diff > angle_value[len(args.label_for_angle)]:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                }
                # torch.save(checkpoint, f'./results/{args.wandb_name}/best_angle.pth')
                best_angle_diff = angle_value[len(args.label_for_angle)]
                best_angle_list = angle_list

        if args.wandb and args.label_for_angle != []:
            log_results_with_angle(
                loss, pixel_loss, angle_loss, dice, 
                rmse_mean, best_rmse_mean, rmse_mean_by_label, best_angle_diff, angle_value,
                len(train_loader), len(val_loader), len(args.label_for_angle)
            )
        elif args.wandb and args.label_for_angle == []:
            log_results(
                loss, pixel_loss, dice, 
                rmse_mean, best_rmse_mean, rmse_mean_by_label,
                len(train_loader), len(val_loader)
            )

    log_terminal(args, 'val_rmse', best_rmse_list)
    log_terminal(args, 'val_angle', best_angle_list)

    return checkpoint