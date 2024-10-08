
import os.path
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt
# from utils.general import plot_confusion_matrix
from models.vgg16custom import vgg
from models.yolov1_classify import Yolov1
from utils.general import ManagerDataYaml, plot_confusion_matrix, ManageSaveDir, save_plots_from_tensorboard
from utils.dataloader import CustomDataLoader
<<<<<<< HEAD
from utils.loss import CrossEntropyLoss, FocalLoss


=======
from utils.loss import CrossEntropyLoss
>>>>>>> dad05f70aa7af8cf9411788d1e04347b9f6b82bd
from utils.metrics import  calculate_accuracy, calculate_precision_recall, confusion_matrix 
import warnings
from utils.augmentations import transform_labels_to_one_hot
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Train YOLOv1 classify from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset', default= '/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size')
    parser.add_argument("--image_size", '-i', type = int, default= 224)
    parser.add_argument('--epochs', '-e', type= int, default= 100)
    parser.add_argument('--learning_rate', '-l', type= float, default= 1e-4)
    parser.add_argument('--resume', action='store_true', help='True if want to resume training')
    parser.add_argument('--pretrain', action='store_true', help='True if want to use pre-trained weights')
<<<<<<< HEAD
    parser.add_argument('--stop_mse_loss', '-st', type= int, default= 0, help= ' num epochs to stop mse loss and change to Cross Entropy loss')
=======
    parser.add_argument('--stop_mse_loss', '-st', type= int, default= 15, help= ' num epochs to stop mse loss and change to Cross Entropy loss')
>>>>>>> dad05f70aa7af8cf9411788d1e04347b9f6b82bd
    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data_yaml_manage = ManagerDataYaml(args.data_yaml)
    data_yaml_manage.load_yaml()
    pretrain_weight = data_yaml_manage.get_properties(key='pretrain_weight')
    categories = data_yaml_manage.get_properties(key='categories')
    num_classes = data_yaml_manage.get_properties(key='num_classes')
<<<<<<< HEAD
    # model =  vgg('A', batch_norm=False, num_classes=num_classes)
    S, B, C = 7, 2 ,3
    model = Yolov1(split_size= S, num_boxes= B, num_classes= C)
    ################################################################################################
    pretrain_weight = '/home/chaos/Documents/ChaosAIVision/temp_folder/yolo_backbone/weights/last.pt'
    checkpoint = torch.load(pretrain_weight)
    checkpoint_state_dict = checkpoint['model_state_dict']

    checkpoint_state_dict.pop('_orig_mod.fcs.1.weight')
    checkpoint_state_dict.pop('_orig_mod.fcs.1.bias')
    model = torch.compile(model)

    model.load_state_dict(checkpoint['model_state_dict'], strict= False)
    ################################################################################################
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum= 0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=1e-2 )
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='sum')

=======
    model =  vgg('A', batch_norm=False, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum= 0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay=1e-2 )
>>>>>>> dad05f70aa7af8cf9411788d1e04347b9f6b82bd

    best_acc = - 100 # create  logic for save weight
    if args.pretrain == True:
        state_dict = torch.load(pretrain_weight)
        model.load_state_dict(state_dict, strict= False)
        print('load weight pretrain sucessfully !')
    if args.resume == True:
        checkpoint = torch.load(pretrain_weight)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epochs = checkpoint['epochs']
        best_acc = checkpoint['best_accuracy']
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print('load weight resume sucessfully !')

    else:
        start_epochs = 0


<<<<<<< HEAD
    # model = torch.compile(model)
=======
    model = torch.compile(model)
>>>>>>> dad05f70aa7af8cf9411788d1e04347b9f6b82bd
    model.to(device)
    train_dataloader = CustomDataLoader(args.data_yaml,'train', args.batch_size, num_workers= 4).create_dataloader()
    valid_loader = CustomDataLoader(args.data_yaml,'valid', args.batch_size, num_workers= 2).create_dataloader()
    locate_save_dir = ManageSaveDir(args.data_yaml)
    weights_folder , tensorboard_folder =  locate_save_dir.create_save_dir() # lấy địa chỉ lưu weight và log
    save_dir = locate_save_dir.get_save_dir_path()
    locate_save_dir.plot_dataset() # plot distribution of dataset
    writer = SummaryWriter(tensorboard_folder)
    mse_loss = torch.nn.MSELoss(reduction= 'sum')
    scaler = torch.cuda.amp.GradScaler()
    # TRAIN
    print(f'result wil save at {save_dir}')
    for epoch in range(start_epochs, args.epochs):
        model.train()
        all_train_losses = []
        all_train_labels = []
        all_train_predictions = []
        progress_bar = tqdm(train_dataloader, colour=  'green')
        for i, (images, labels) in enumerate(progress_bar):
            # labels =  transform_labels_to_one_hot(labels,num_classes )
            images = images.to(device)
            labels = labels.to(device)
            if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)) or \
           torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                continue
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output =  model(images)
                prediction_train = torch.argmax(output, dim= 1)
                interger_labels = torch.argmax(labels, dim= 1)
                if epoch < args.stop_mse_loss: # Use loss MSE when start trainning help model optimizer bettter 
                    output = output.float()
                    labels = labels.float()

                    loss = mse_loss (output, labels)
                else:
<<<<<<< HEAD
                    loss = focal_loss(output, labels)
=======
                    loss = CrossEntropyLoss(output, labels)
>>>>>>> dad05f70aa7af8cf9411788d1e04347b9f6b82bd
            all_train_losses.append(loss.item())
            all_train_labels.extend(interger_labels.tolist())
            all_train_predictions.extend(prediction_train.tolist())
            progress_bar.set_description(f"Epochs {epoch + 1}/{args.epochs} loss: {loss :0.4f}")
            # writer.add_scalar('Train/loss', loss, epoch * len(train_dataloader) + i)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        avagare__train_loss = np.mean(all_train_losses)
        accuracy_train  = calculate_accuracy(all_train_labels, all_train_predictions, is_all= True)
        cm_train = confusion_matrix(all_train_labels, all_train_predictions)
        precision_recall_train = calculate_precision_recall(cm_train, categories, 'all')
        writer.add_scalar("Train/mean_loss", avagare__train_loss, epoch)
        writer.add_scalar("Train/accuracy", accuracy_train, epoch)
        writer.add_scalar("Train/precision", precision_recall_train['average_precision'], epoch)
        writer.add_scalar("Train/recall", precision_recall_train['average_recall'], epoch)



    
    # VALIDATION

        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            progress_bar = tqdm(valid_loader, colour=  'yellow')
            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                if torch.any(torch.isnan(images)) or torch.any(torch.isinf(images)) or \
                torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                    continue
                output = model(images)
                with torch.cuda.amp.autocast():
                    prediction = torch.argmax(output, dim= 1)
                    interger_labels = torch.argmax(labels, dim= 1)
                    if epoch < args.stop_mse_loss: # Use loss MSE when start trainning help model optimizer bettter 
                        output = output.float()
                        labels = labels.float()
                        loss = mse_loss (output, labels)
                    else:
<<<<<<< HEAD
                        loss = focal_loss(output, labels)
=======
                        loss = CrossEntropyLoss(output, labels)
>>>>>>> dad05f70aa7af8cf9411788d1e04347b9f6b82bd
                progress_bar.set_description(f"Epochs {epoch + 1}/{args.epochs} loss: {loss :0.4f}")
                all_losses.append(loss.item())
                all_labels.extend(interger_labels.tolist())
                all_predictions.extend(prediction.tolist())
                # writer.add_scalar('Valid/loss', loss, epoch * len(valid_loader) + i)


            avagare_loss = np.mean(all_losses)
            accuracy  = calculate_accuracy(all_labels, all_predictions, is_all= True)
            cm = confusion_matrix(all_labels, all_predictions)
            precision_recall = calculate_precision_recall(cm, categories, 'all')
            print(f"precision: {precision_recall['average_precision' ] :0.4f}  recall: {precision_recall['average_recall']:0.4f} loss: {avagare_loss :0.4f} accuracy: {accuracy :0.4f}")
            writer.add_scalar("Valid/accuracy", accuracy, epoch)
            writer.add_scalar("Valid/mean_loss", avagare_loss, epoch)
            writer.add_scalar("Valid/precision", precision_recall['average_precision'], epoch)
            writer.add_scalar("Valid/recall", precision_recall['average_recall'], epoch)
            plot_confusion_matrix(writer, cm, categories, epoch)
            # checkpoint = {
            #     'model_state_dict': model.state_dict(),
            #     'epochs' : epoch,
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'best_accuracy': best_acc
            # }
            checkpoint = {
                'model_state_dict': model.state_dict()}
            torch.save(checkpoint,os.path.join( weights_folder, 'last.pt'))
            if accuracy > best_acc:
                torch.save(checkpoint,os.path.join( weights_folder, 'best.pt'))
                best_acc = accuracy

    save_plots_from_tensorboard(tensorboard_folder, save_dir)    


        


if __name__ == "__main__":
    args = get_args()
    data_yaml = args.data_yaml
    train(args)
