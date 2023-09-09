import torch
from PIL import Image
import torchvision.transforms as transforms
import os

from preprocess_image.preprocess_image import rec_digit
from load_model import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_net = load_model().to(device)

curr_dir = os.getcwd()

for image_file in os.listdir(os.path.join(curr_dir, "my_images")):
    image = rec_digit(Image.open(os.path.join(curr_dir, "my_images", image_file)))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    input_image = transform(image).unsqueeze(0)  # unsqueeze добавляет размерность батча (1), так как ожидается батч

    input_image = input_image.to(device)
    output = loaded_net(input_image)
    predicted_class = torch.argmax(output).item()
    print(f'Картинка: {image_file} Предсказанный класс: {predicted_class}')
