import argparse
from models.common import ManagerDataYaml, ModelManagement
import torch.nn as nn
import traceback
import torch
import cv2
import numpy as np
import os
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="Train VGG16 from scratch")
    parser.add_argument('--data_yaml', "-d", type= str, help= 'Path to dataset')
    parser.add_argument('--batch_size', '-b', type = int, help = 'input batch_size', default= None)
    parser.add_argument("--image_size", '-i', type = int, default= 224)
    parser.add_argument("--batch_norm", type = bool, default= False)
    parser.add_argument("--source", type = str)
    parser.add_argument("--weights", type = str)
    parser.add_argument("--model_name", '-name', type = str)
    parser.add_argument("--format",  type = str, help= " support pt is pytorch, onnx and tensorrt")
    parser.add_argument("--version", '-v', type = str)
    parser.add_argument("--save_dir", type = str, default= run_path)
    parser.add_argument('--save', action= 'store_true'  )
    parser.add_argument('--plot', action= 'store_true'  )
    parser.add_argument("--task", type = str, help= ' task = classify or task = detect')



    return parser.parse_args()  # Cần trả về kết quả từ parser.parse_args()


def inferences_frame(args, model, classes, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tiền xử lý ảnh
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (args.image_size, args.image_size))
    image_normalized = (image_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(np.transpose(image_normalized, (2, 0, 1))[None, :, :, :]).float().to(device)

    # Dự đoán
    model.to(device).eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = nn.Softmax(dim=1)(output)
        predicted_prob, predicted_class = torch.max(prob, dim=1)
        score = predicted_prob[0].item() * 100
        label = classes[predicted_class[0].item()]

    # Vẽ nhãn và điểm tin cậy lên ảnh
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    label_text = f"{label} with confidence score of {score:.2f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label_text, font, 1, 2)[0]
    text_x = (image_bgr.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10
    cv2.putText(image_bgr, label_text, (text_x, text_y), font, 1, (0, 0, 255), 2)

    return image_bgr, score, label


class Detect():
    def __init__(self, args):
        model_management = ModelManagement(args.model_name, args.version, args.weights, args.data_yaml, args.batch_norm)
        try:
            self.model = model_management.loading_weight()
        except Exception as e:
            print("An error occurred while loading the model weight:")
            print(str(e))
            # Optionally, you can print the traceback to get more details
            traceback.print_exc()

        self.image = None
        self.video = None
        self.webcam = None
        data_yaml_manage = ManagerDataYaml(args.data_yaml)
        data_yaml_manage.load_yaml()
        self.classes = data_yaml_manage.get_properties('categories')
        self.save_dir = args.save_dir
        

    def read_image(self):
        self.image = cv2.imread(args.source)


    def save_image(self):
        if self.image is None:
            print("No image to save.")
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)        
        base_name = os.path.basename(args.source)
        name, ext = os.path.splitext(base_name)
        new_file_name = f"{name}_processed{ext}"
        save_path = os.path.join(self.save_dir, new_file_name)

        # Lưu ảnh
        cv2.imwrite(save_path, self.image)
        print(f"Image saved to {save_path}")

    def plot_image(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        pil_image.show()

    
    def process_image(self):
        self.read_image()
        self.image, _, _ = inferences_frame(args, self.model, self.classes, self.image)
        if args.save:
            self.save_image()
        if args.plot:
            self.plot_image()




    




if __name__ == ("__main__"):
    run_path = os.path.join(os.getcwd(), 'runs')
    args = get_args()
    detect  = Detect(args)
    process_image = detect.process_image()


