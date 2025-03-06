import os
import torch
from torchvision import transforms, models
from PIL import Image

training_model = "trained_model18_202.pth"
plazma_do_spawania = 0
plazma_do_napawania = 0
TIG = 0
GMA = 0
GMA_2 = 0

photo_size = (256, 256)

model = models.resnet18()
number_class = 5
model.fc = torch.nn.Linear(model.fc.in_features, number_class)
model.load_state_dict(torch.load(training_model))
model.eval()

transform = transforms.Compose([
    transforms.Resize(photo_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

for i in range(1, 51):
    photo_name = f"{i}.jpg"
    path = os.path.join("glowice_sprawdz", photo_name)

    if os.path.exists(path):
        photo = Image.open(path)

        photo = transform(photo)
        photo = photo.unsqueeze(0)

        with torch.no_grad():
            output = model(photo)

        _, check = torch.max(output, 1)

        if check.item() == 0:
            predicted_class = 'plazma_do_spawania'
            plazma_do_spawania += 1
        elif check.item() == 1:
            predicted_class = 'plazma_do_napawania'
            plazma_do_napawania += 1
        elif check.item() == 2:
            predicted_class = 'TIG'
            TIG += 1
        elif check.item() == 3:
            predicted_class = 'GMA'
            GMA += 1
        elif check.item() == 4:
            predicted_class = 'GMA_2'
            GMA_2 += 1

        print(f"Zdjęcie {photo_name}: {predicted_class}")
        
    else:
        print(f"Zdjęcie {photo_name} nie istnieje.")

print("Liczba zdjęć klasy 'plazma_do_spawania':", plazma_do_spawania)
print("Liczba zdjęć klasy 'plazma_do_napawania':", plazma_do_napawania)
print("Liczba zdjęć klasy 'TIG':", TIG)
print("Liczba zdjęć klasy 'GMA':", GMA)
print("Liczba zdjęć klasy 'GMA_2':", GMA_2)