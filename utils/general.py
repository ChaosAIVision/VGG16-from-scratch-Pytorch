import yaml
import os 
import sys
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from train import data_yaml

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

    def get_properties(self, key: str) -> str:
        """
        Get the value of a specific property from the loaded YAML data.
        """
        if isinstance(self.data, dict):
            if key in self.data:
                value = self.data[key]
                return str(value)
            else:
                return f"Key '{key}' not found in the data."
        else:
            return "Data has not been loaded or is not a dictionary."
        

class ManageSaveDir():
    def __init__(self):
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
                        break
                    counter += 1
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
        for categories in self.categories:
            categories_path = os.path.join(data_path, categories)
            num_labels = self.count_items_in_folder(categories_path)
            num_categories.append(num_labels)
        return num_categories

    def plot_dataset(self):
            # Lấy số lượng hình ảnh cho các tập dữ liệu
            distribution_train = self.count_distribution_labels('train')
            distribution_valid = self.count_distribution_labels('valid')
            
            num_image_train = sum(distribution_train)
            num_image_valid = sum(distribution_valid)
            if self.test_dataset is not None:
                distribution_test = self.count_distribution_labels('test')
                num_image_test = sum(distribution_valid)
            else:
                distribution_test = 0
                num_image_test = 0


            
            file_path = os.path.join(self.result_dir, 'dataset_distribution.png')

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))

            ax[0].axis('off')
            text_str = f"Number of images in train: {num_image_train}\n" \
                    f"Number of images in valid: {num_image_valid}\n" \
                    f"Number of images in test: {num_image_test}"
            ax[0].text(0.5, 0.5, text_str, fontsize=12, ha='center', va='center')

            x = range(len(self.categories))
            width = 0.2

            ax[1].bar([i - width for i in x], distribution_train, width=width, label='Train', align='center')
            ax[1].bar(x, distribution_valid, width=width, label='Valid', align='center')
            ax[1].bar([i + width for i in x], distribution_test, width=width, label='Test', align='center')

            ax[1].set_xlabel('Categories')
            ax[1].set_ylabel('Number of images')
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(self.categories, rotation=90)
            ax[1].legend()

            plt.tight_layout()

            # Lưu hình ảnh vào tệp tin
            plt.savefig(file_path)
            plt.close()  # Đóng hình ảnh để giải phóng bộ nhớ





    








if __name__ == "__main__":
    # Create an instance of ManagerDataYaml
    # manager = ManagerDataYaml('/Users/chaos/Documents/Chaos_working/Chaos_projects/VGG16-from-scratch-Pytorch/data.yaml')
    
    # # Load YAML data
    # data = manager.load_yaml()
    
    # # Get specific property
    # properties = manager.get_properties('train')
    
    # # Print results
    # print(data)
    # print(properties)
    create_save_dir = ManageSaveDir()
    create_save_dir.create_save_dir()


