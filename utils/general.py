import yaml
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
import io
import os 
import sys
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import numpy as np
class ManagerDataYaml:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = None

    def load_yaml(self) -> dict:
        """
        Load data from YAML file and return its properties as a dictionary.
        """
        try:
            with open(self.yaml_path, 'r') as file:
                self.data = yaml.safe_load(file)
                return self.data
        except Exception as e:
            return f"Error loading YAML file: {self.yaml_path}. Exception: {e}"

    def get_properties(self, key: str) :
        """
        Get the value of a specific property from the loaded YAML data.
        """
        if isinstance(self.data, dict):
            if key in self.data:
                value = self.data[key]
                return (value)
            else:
                return f"Key '{key}' not found in the data."
        else:
            return "Data has not been loaded or is not a dictionary."
        

class ManageSaveDir():
    def __init__(self, data_yaml):
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        self.save_dir_locations = data_yaml_manage.get_properties('save_dirs')
        self.train_dataset = data_yaml_manage.get_properties('train')
        self.valid_dataset = data_yaml_manage.get_properties('valid')
        self.test_dataset = data_yaml_manage.get_properties('test')
        self.categories = data_yaml_manage.get_properties('categories')

    def create_save_dir(self):
        if not os.path.exists(self.save_dir_locations):
            return f'Folder path {self.save_dir_locations} is not exists'
        else:
            self.result_dir = os.path.join(self.save_dir_locations, 'result')
            weight_dir = os.path.join(self.result_dir, 'weights')
            tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')

            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
                os.makedirs(weight_dir)
                os.makedirs(tensorboard_dir)
                return weight_dir, tensorboard_dir # Cần return tensorbard_dir để lấy location  ghi log và weight
            else:
                counter = 1
                while True:
                    self.result_dir = os.path.join(self.save_dir_locations, f'result{counter}')
                    weight_dir = os.path.join(self.result_dir, 'weights')
                    tensorboard_dir = os.path.join(self.result_dir, 'tensorboard')
                    if not os.path.exists(self.result_dir):
                        os.makedirs(self.result_dir)
                        os.makedirs(weight_dir)
                        os.makedirs(tensorboard_dir)
                        return weight_dir, tensorboard_dir
                    counter += 1
    def get_save_dir_path(self):
        return self.result_dir
    def count_items_in_folder(self, folder_path):
        try:
            items = os.listdir(folder_path)
            num_items = len(items)
            return num_items
        except FileNotFoundError:
            return f'The folder {folder_path} does not exist'
        except PermissionError:
            return f'Permission denied to access the folder {folder_path}'
        
    def count_distribution_labels(self, mode):
        if mode == 'train':
            data_path = self.train_dataset
        elif mode == 'valid':
            data_path = self.valid_dataset
        else:
            data_path = self.test_dataset

        num_categories = []
        for category in self.categories:
            categories_path = os.path.join(data_path, category)
            num_labels = self.count_items_in_folder(categories_path)
            num_categories.append(num_labels)
        return num_categories



    def plot_dataset(self):
        # Lấy số lượng hình ảnh cho các tập dữ liệu
        distribution_train = self.count_distribution_labels('train')
        distribution_valid = self.count_distribution_labels('valid')
        
        num_image_train = sum(distribution_train)
        num_image_valid = sum(distribution_valid)
        try:
            distribution_test = self.count_distribution_labels('test')
            num_image_test = sum(distribution_test)
        except:
            distribution_test = [0] * len(self.categories)  # Sửa tại đây
            num_image_test = 0

        # Tổng hợp số lượng hình ảnh cho mỗi danh mục
        total_distribution = [0] * len(self.categories)
        
        for dist in (distribution_train, distribution_valid, distribution_test):
            for i, count in enumerate(dist):
                if isinstance(count, int):  # Đảm bảo giá trị là số nguyên
                    total_distribution[i] += count
        
        file_path = os.path.join(self.result_dir, 'dataset_distribution.png')

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # Plot thông tin tổng số hình ảnh
        ax[0].axis('off')
        text_str = f"Number of images in train: {num_image_train}\n" \
                f"Number of images in valid: {num_image_valid}\n" \
                f"Number of images in test: {num_image_test}"
        ax[0].text(0.5, 0.5, text_str, fontsize=12, ha='center', va='center')

        # Plot số lượng hình ảnh tổng hợp cho từng danh mục
        x = range(len(self.categories))
        width = 0.2

        # Tạo màu cho từng danh mục
        cmap = plt.get_cmap('tab20')  # Lấy bảng màu 'tab20'
        colors = [cmap(i) for i in range(len(self.categories))]

        # Vẽ biểu đồ cột với từng màu khác nhau
        ax[1].bar(x, total_distribution, color=colors, width=width, label='Total', align='center')

        ax[1].set_xlabel('Categories')
        ax[1].set_ylabel('Number of images')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(self.categories, rotation=90)
        ax[1].legend()

        plt.tight_layout()

        # Lưu hình ảnh vào tệp tin
        plt.savefig(file_path)
        plt.close()  # Đóng hình ảnh để giải phóng bộ nhớ
            
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=14)
    plt.yticks(tick_marks, class_names, fontsize=14)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, f'{cm[i, j]:.2f}', horizontalalignment="center", color=color,
                     fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    writer.add_figure('confusion_matrix', figure, epoch)



import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def save_plots_from_tensorboard(tensorboard_folder, output_image_folder):
    # Khởi tạo EventAccumulator để đọc các tệp sự kiện trong thư mục
    event_accumulator = EventAccumulator(tensorboard_folder)
    event_accumulator.Reload()

    # Lấy tất cả các tags từ TensorBoard
    scalar_tags = event_accumulator.Tags()['scalars']
    
    # Lấy dữ liệu cho các scalar tags
    def get_scalar_data(tag):
        events = event_accumulator.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        return steps, values

    # Tạo tấm ảnh đầu tiên với 8 biểu đồ
    plt.figure(figsize=(16, 8))
    
    # Các tag cho huấn luyện
    train_tags = ['Train/loss', 'Train/accuracy', 'Train/precision', 'Train/recall']
    # Các tag cho kiểm tra
    valid_tags = ['Valid/loss', 'Valid/accuracy', 'Valid/precision', 'Valid/recall']

    # Vẽ các biểu đồ cho tập huấn luyện
    for i, tag in enumerate(train_tags):
        plt.subplot(2, 4, i + 1)
        steps, values = get_scalar_data(tag)
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')

    # Vẽ các biểu đồ cho tập kiểm tra
    for i, tag in enumerate(valid_tags):
        plt.subplot(2, 4, i + 5)
        steps, values = get_scalar_data(tag)
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel('Steps')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig(os.path.join(output_image_folder, 'training_and_validation_plots.png'))
    plt.close()

    # Lấy dữ liệu ma trận nhầm lẫn
    confusion_matrix_tag = 'confusion_matrix'
    image_events = event_accumulator.Images(confusion_matrix_tag)

    if image_events:
        # Lấy bước cuối cùng từ các sự kiện hình ảnh
        last_step = max(event.step for event in image_events)
        
        # Lọc sự kiện hình ảnh với bước cuối cùng
        for event in reversed(image_events):
            if event.step == last_step:
                image_string = event.encoded_image_string
                image = Image.open(io.BytesIO(image_string))
                
                plt.figure(figsize=(8, 8))
                plt.imshow(image)
                plt.title('Confusion Matrix')
                plt.axis('off')
                plt.savefig(os.path.join(output_image_folder, 'confusion_matrix.png'))
                plt.close()
                break
    else:
        print("No confusion matrix found in TensorBoard logs.")
