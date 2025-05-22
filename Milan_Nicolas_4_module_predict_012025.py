import torch
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_images_from_folder(folder_path):
    images = []
    file_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)
            file_names.append(filename)
    return images, file_names


def predict_image(image_path, model, labels):
    image = Image.open(image_path).convert('RGB')
    input_tensor = data_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_label = labels[predicted.item()]
    return predicted_label


def main(folder_path, labels, model):
    images, file_names = load_images_from_folder(folder_path)

    for img_path, file_name in zip(images, file_names):
        predicted_label = predict_image(img_path, model, labels)
        image = Image.open(img_path)
        plt.imshow(image)
        plt.title(f"Fichier: {file_name}\nPrédiction: {predicted_label}")
        plt.axis('off')
        plt.show()


def load_labels(label_path):
    with open(label_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def load_image_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Le dossier {folder_path} n'existe pas.")
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))]


def load_model(model_path):
    return torch.load(model_path, map_location=torch.device(device))


if __name__ == "__main__":
    model_path = input("Entrez le chemin du modèle: ")
    model = models.resnet18(pretrained=False)
    num_classes = 120
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device(device)))
    model.eval()
    folder_path = input("Entrez le chemin du dossier contenant les images: ")
    label_path = input("Le path de la liste de labels: ")
    with open(label_path, 'r') as file:
        labels = file.read().strip().split(',')
    print(len(labels))

    main(folder_path, labels, model)
