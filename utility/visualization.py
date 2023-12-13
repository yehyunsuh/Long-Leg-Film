import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont


def visualize(
        args, idx, image_path, image_name, label, label_list, epoch, 
        extracted_pixels_list, prediction, prediction_binary,
        predict_spatial_mean, label_spatial_mean, angles, mode
    ):
    original_image= Image.open(image_path).resize((args.image_resize,args.image_resize)).convert("RGB")

    if mode == 'train':
        if epoch % 50 == 0:
            image_w_label(args, image_name, original_image, label, label_list, epoch, idx)
        if idx == 0:
            image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list, mode)
            angle_visualization(args, idx, image_name, original_image, extracted_pixels_list, epoch, angles)
            # image_w_seg_pred(args, idx, image_name, original_image, epoch, prediction_binary, predict_spatial_mean, label_spatial_mean)
            # image_w_heatmap(args, idx, image_name, epoch, prediction)
    else:
        image_w_ground_truth_and_prediction(args, idx, image_name, original_image, None, extracted_pixels_list, label_list, mode)


def image_w_label(args, image_name, original_image, label, label_list, epoch, idx):
    label_image = original_image.copy()
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])
        
        if y != 0 and x != 0:
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))

    original_image.save(f'{args.result_directory}/{args.wandb_name}/label/{image_name}_label.png')

    if idx == 0:
        for i in range(args.output_channel):
            background = label[0][i].unsqueeze(0)
            background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
            overlaid_image = Image.blend(label_image, background , 0.3)
            overlaid_image.save(f'{args.result_directory}/{args.wandb_name}/label/{image_name}_{epoch}_label{i}.png')
            break


def image_w_ground_truth_and_prediction(args, idx, image_name, original_image, epoch, extracted_pixels_list, label_list, mode):
    for i in range(args.output_channel):
        y = int(label_list[2*i])
        x = int(label_list[2*i+1])
        if y != 0 and x != 0:
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (0, 0, 255),-1))
            y = int(extracted_pixels_list[idx][0][i][0])
            x = int(extracted_pixels_list[idx][0][i][1])
            original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), 8, (255, 0, 0),-1))

    if mode == 'test':
        original_image.save(f'{args.result_directory}/{args.wandb_name}/test/{image_name}.png')
    else:
        original_image.save(f'{args.result_directory}/{args.wandb_name}/pred_w_gt/{image_name}_{epoch}.png')


def image_w_seg_pred(args, idx, image_name, original_image, epoch, prediction_binary, predict_spatial_mean, label_spatial_mean):
    for i in range(args.output_channel):
        background = prediction_binary[0][i].unsqueeze(0)
        background = TF.to_pil_image(torch.cat((background, background, background), dim=0))
        overlaid_image = Image.blend(original_image, background , 0.3)
        x = int(label_spatial_mean[0][i][0])
        y = int(label_spatial_mean[0][i][1])
        overlaid_image = Image.fromarray(cv2.circle(np.array(overlaid_image), (x,y), 8, (0, 0, 255),-1))

        x = int(predict_spatial_mean[0][0][0])
        y = int(predict_spatial_mean[0][0][1])
        overlaid_image = Image.fromarray(cv2.circle(np.array(overlaid_image), (x,y), 8, (255, 0, 0),-1))
        overlaid_image.save(f'{args.result_directory}/{args.wandb_name}/heatmap/label{i}/{image_name}_{epoch}_label{i}.png')


def image_w_heatmap(args, idx, image_name, epoch, prediction):
    for i in range(len(prediction[0])):
        plt.imshow(prediction[0][i].detach().cpu().numpy(), interpolation='nearest')
        plt.axis('off')
        plt.savefig(f'./results/{args.wandb_name}/heatmap/label{i}/{image_name}_{epoch}_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=150)


def intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4):
    if x2-x1 == 0: m1 = (y2-y1)/1e-8
    else:          m1 = (y2-y1)/(x2-x1)
    if x4-x3 == 0: m2 = (y4-y3)/1e-8
    else:          m2 = (y4-y3)/(x4-x3)

    if m1 == m2:
        # print("두 선이 평행합니다")
        return None, None
    else:
        x_intersect = (m1*x1-y1-m2*x3+y3)/(m1-m2)
        y_intersect = m1*(x_intersect-x1)+y1
        return x_intersect, y_intersect


def angle_visualization(args, idx, image_name, original_image, extracted_pixels_list, epoch, angles):
    line_width, circle_size = 3, 4
    LDFA_text = f'LDFA: {angles[0]:.2f}\nAnswer: {angles[3]:.2f}'
    MPTA_text = f'MPTA: {angles[1]:.2f}\nAnswer: {angles[4]:.2f}'
    mHKA_text = f'MHKA: {angles[2]:.2f}\nAnswer: {angles[5]:.2f}'
    red, green, blue = (255, 0, 0), (0,102,0), (0, 0, 255)

    rgb = [(255, 0, 0), (0,102,0), (0, 0, 255)]
    text = [LDFA_text, MPTA_text, mHKA_text]

    count, pixels = 0, []
    font = ImageFont.truetype("data/font/Gidole-Regular.ttf", size=20)

    for i in range(args.output_channel):
        x, y = int(extracted_pixels_list[idx][0][i][1]), int(extracted_pixels_list[idx][0][i][0])
        pixels.append([x,y])

        original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), circle_size, red,-1))
       
        # if count == 2: pass
        # elif count <= 3: original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), circle_size, red,-1))
        # elif count == 5: pass
        # else:          original_image = Image.fromarray(cv2.circle(np.array(original_image), (x,y), circle_size, blue,-1))
        # count += 1

    intersection = []
    x1, y1, x2, y2 = pixels[3][0], pixels[3][1], pixels[5][0], pixels[5][1]
    x3, y3, x4, y4 = pixels[2][0], pixels[2][1], pixels[4][0], pixels[4][1]
    intersection.append(intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4))
    # x5, y5 = intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4)
    # print(x5, y5)

    x1, y1, x2, y2 = pixels[6][0], pixels[6][1], pixels[8][0], pixels[8][1]
    x3, y3, x4, y4 = pixels[7][0], pixels[7][1], pixels[9][0], pixels[9][1]
    intersection.append(intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4))
    # x5, y5 = intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4)
    # print(x5, y5)

    x1, y1, x2, y2 = pixels[1][0], pixels[1][1], pixels[4][0], pixels[4][1]
    x3, y3, x4, y4 = pixels[7][0], pixels[7][1], pixels[10][0], pixels[10][1]
    intersection.append(intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4))
    # x5, y5 = intersection_line_to_line(x1, y1, x2, y2, x3, y3, x4, y4)
    # print(x5, y5)
    # exit()
    
    draw = ImageDraw.Draw(original_image)
    # LDFA
    line1 = ((pixels[3][0],pixels[3][1]),(pixels[5][0],pixels[5][1]))
    line2 = ((pixels[2][0],pixels[2][1]),(intersection[0][0],intersection[0][1]))
    # MPTA
    line3 = ((pixels[6][0],pixels[6][1]),(pixels[8][0],pixels[8][1]))
    line4 = ((pixels[9][0],pixels[9][1]),(intersection[1][0],intersection[1][1]))
    # MHKA
    line5 = ((pixels[1][0],pixels[1][1]),(intersection[2][0],intersection[2][1]))
    line6 = ((pixels[7][0],pixels[7][1]),(pixels[10][0],pixels[10][1]))
    line_pixel = [line1, line2, line3, line4, line5, line6]

    text_pixel = [
        (((pixels[2][0]+intersection[0][0])/2-150), (2*pixels[2][1]+5*intersection[0][1])/7),
        (((pixels[9][0]+intersection[1][0])/2-150), (4*pixels[9][1]+intersection[1][1])/5),
        (((pixels[1][0]+intersection[2][0])/2+50), (pixels[1][1]+intersection[2][1])/2),   
    ]

    draw = draw_line(draw, line_pixel, rgb, line_width, pixels)
    draw = draw_text(draw, text_pixel, text, rgb, font)
    original_image.save(f'{args.result_directory}/{args.wandb_name}/pred_w_gt/{image_name}_{epoch}_angle.png')


def draw_line(draw, line_pixel, rgb, line_width, pixels):
    draw.line(line_pixel[0], fill=rgb[0], width=line_width)
    draw.line(line_pixel[1], fill=rgb[0], width=line_width)
    draw.line(line_pixel[2], fill=rgb[2], width=line_width)
    draw.line(line_pixel[3], fill=rgb[2], width=line_width)
    draw.line(line_pixel[4], fill=rgb[1], width=line_width)
    draw.line(line_pixel[5], fill=rgb[1], width=line_width)
    return draw


def draw_text(draw, text_pixel, text, rgb, font):
    draw.text(text_pixel[0], text[0], fill=rgb[0], align ="left", font=font) 
    draw.text(text_pixel[1], text[1], fill=rgb[2], align ="left", font=font) 
    draw.text(text_pixel[2], text[2], fill=rgb[1], align ="left", font=font) 
    return draw