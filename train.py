
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
from utils.general import ManagerDataYaml, plot_confusion_matrix, ManageSaveDir
from utils.dataloader import CustomDataLoader
from utils.loss import CrossEntropyLoss
from utils.metrics import  calculate_accuracy, calculate_precision_recall, confusion_matrix 

# warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Train VGG16 from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset', default= '/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size')
    parser.add_argument("--image_size", '-i', type = int, default= 224)
    parser.add_argument('--epochs', '-e', type= int, default= 100)
    parser.add_argument('--learning_rate', '-l', default= 1e-2)
    parser.add_argument('--pretrain_weight', '-pr')
    parser.add_argument('--resume', type= bool, default= False, help= 'True if want to resume training')
    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()


def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data_yaml_manage = ManagerDataYaml(args.data_yaml)
    data_yaml_manage.load_yaml()
    pretrain_weight = data_yaml_manage.get_properties(key='pretrain_weight')
    categories = data_yaml_manage.get_properties(key='categories')
    model =  vgg('D', batch_norm=False, num_classes=2)
    state_dict = torch.load(pretrain_weight)
    model.load_state_dict(state_dict, strict= False)
    model.to(device)

    train_dataloader = CustomDataLoader(args.data_yaml,'train', args.batch_size, num_workers= 6)
    valid_loader = CustomDataLoader(args.data_yaml,'valid', args.batch_size, num_workers= 6)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
    locate_save_dir = ManageSaveDir(args.data_yaml)
    weights_folder , tensorboard_folder =  locate_save_dir.create_save_dir() # lấy địa chỉ lưu weight và log
    locate_save_dir.plot_dataset() # plot distribution of dataset
    writer = SummaryWriter(tensorboard_folder)
    best_acc = - 100 # create  logic for save weight

    # TRAIN
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour=  'green')
        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output =  model(images)
            loss = CrossEntropyLoss(output, labels)
            progress_bar.set_description(f'Epochs {epoch + 1} / {args.epochs}', loss)
            writer.add_scalar('Train/loss', loss, epoch * len(train_dataloader) + i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # VALIDATION

        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            progress_bar = tqdm(train_dataloader, colour=  'yellow')
            for i, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)

                prediction = torch.argmax(output, dim= 1)
                loss = CrossEntropyLoss(output, labels)
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(prediction.tolist())
            avagare_loss = np.mean(all_losses)
            accuracy  = calculate_accuracy(all_labels, all_predictions, is_all= False)
            cm = confusion_matrix(all_labels, all_predictions)
            precision_recall = calculate_precision_recall(cm, categories, 'all')
            print(f'precision: {precision_recall['average_precision']} recall: {precision_recall['average_recall']} loss:{avagare_loss} accuracy: {accuracy}')
            writer.add_scalar("Valid/loss", avagare_loss, epoch)
            writer.add_scalar("Valid/accuracy", accuracy, epoch)
            writer.add_scalar("Valid/precision", precision_recall['average_precision'], epoch)
            writer.add_scalar("Valid/recall", precision_recall['average_recall'], epoch)
            plot_confusion_matrix(writer, cm, categories, epoch)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epochs' : epoch,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, weights_folder, 'last.pt')
            if accuracy > best_acc:
                torch.save(checkpoint, weights_folder, 'best.pt')
                best_acc = accuracy

            


        







if __name__ == "__main__":
    args = get_args()
    data_yaml = args.data_yaml
    train(args)
