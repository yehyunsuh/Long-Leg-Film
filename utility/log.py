import wandb


def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results(
    loss, pixel_loss, dice, 
    rmse_mean, best_rmse_mean, rmse_mean_by_label,
    train_loader_len, val_loader_len
    ):
    wandb.log({
        'Train Loss': loss/train_loader_len,
        'Train Pixel Loss': pixel_loss/train_loader_len,
        'Dice Score': dice,
        'Mean RMSE': rmse_mean,
        'Best Mean RMSE': best_rmse_mean,
        'Label1': rmse_mean_by_label[0],
        'Label2': rmse_mean_by_label[1],
        'Label3': rmse_mean_by_label[2],
        'Label4': rmse_mean_by_label[3],
        'Label5': rmse_mean_by_label[4],
        'Label6': rmse_mean_by_label[5],
        'Label7': rmse_mean_by_label[6],
        'Label8': rmse_mean_by_label[7],
        'Label9': rmse_mean_by_label[8],
        'Label10': rmse_mean_by_label[9],
        'Label11': rmse_mean_by_label[10],
    })


def log_test_results(dice, rmse_mean, rmse_mean_by_label):
    wandb.log({
        'Test Dice Score': dice,
        'Test Mean RMSE': rmse_mean,
        'Test Label1': rmse_mean_by_label[0],
        'Test Label2': rmse_mean_by_label[1],
        'Test Label3': rmse_mean_by_label[2],
        'Test Label4': rmse_mean_by_label[3],
        'Test Label5': rmse_mean_by_label[4],
        'Test Label6': rmse_mean_by_label[5],
        'Test Label7': rmse_mean_by_label[6],
        'Test Label8': rmse_mean_by_label[7],
        'Test Label9': rmse_mean_by_label[8],
        'Test Label10': rmse_mean_by_label[9],
        'Test Label11': rmse_mean_by_label[10],
    })


def log_results_with_angle(
    loss, pixel_loss, angle_loss, dice,
    rmse_mean, best_rmse_mean, rmse_mean_by_label, best_angle_diff, angle_value,
    train_loader_len, val_loader_len, angle_len
    ):
    wandb.log({
        'Train Loss': loss/train_loader_len,
        'Train Pixel Loss': pixel_loss/train_loader_len,
        'Train Angle Loss': angle_loss/train_loader_len,
        'Dice Score': dice,
        'Mean RMSE': rmse_mean,
        'Best Mean RMSE': best_rmse_mean,
        'LDFA': angle_value[0]/(val_loader_len),
        'MPTA': angle_value[1]/(val_loader_len),
        'MKHA': angle_value[2]/(val_loader_len),
        'Long MKHA': angle_value[3]/(val_loader_len),
        'Mean Angle Difference': angle_value[4]/(val_loader_len*angle_len),
        'Best Mean Angle Difference': best_angle_diff/(val_loader_len*angle_len),
        'Label1': rmse_mean_by_label[0],
        'Label2': rmse_mean_by_label[1],
        'Label3': rmse_mean_by_label[2],
        'Label4': rmse_mean_by_label[3],
        'Label5': rmse_mean_by_label[4],
        'Label6': rmse_mean_by_label[5],
        'Label7': rmse_mean_by_label[6],
        'Label8': rmse_mean_by_label[7],
        'Label9': rmse_mean_by_label[8],
        'Label10': rmse_mean_by_label[9],
        'Label11': rmse_mean_by_label[10],
    })


def log_test_results_with_angle(dice, rmse_mean, rmse_mean_by_label, angle_value, test_loader_len, angle_len):
    wandb.log({
        'Test Dice Score': dice,
        'Test Mean RMSE': rmse_mean,
        'Test LDFA': angle_value[0]/(test_loader_len),
        'Test MPTA': angle_value[1]/(test_loader_len),
        'Test MKHA': angle_value[2]/(test_loader_len),
        'Test Long MKHA': angle_value[3]/(test_loader_len),
        'Test Mean Angle Difference': angle_value[4]/(test_loader_len*angle_len),
        'Test Label1': rmse_mean_by_label[0],
        'Test Label2': rmse_mean_by_label[1],
        'Test Label3': rmse_mean_by_label[2],
        'Test Label4': rmse_mean_by_label[3],
        'Test Label5': rmse_mean_by_label[4],
        'Test Label6': rmse_mean_by_label[5],
        'Test Label7': rmse_mean_by_label[6],
        'Test Label8': rmse_mean_by_label[7],
        'Test Label9': rmse_mean_by_label[8],
        'Test Label10': rmse_mean_by_label[9],
        'Test Label11': rmse_mean_by_label[10],
    })



def log_terminal(args, data_type, *a):
    file = open(f'{args.result_directory}/{args.wandb_name}/{args.wandb_name}_{data_type}.txt', 'a')
    print(*a, file=file)