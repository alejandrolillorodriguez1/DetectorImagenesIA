import torch
import torchvision.datasets as datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

total_data = datasets.ImageFolder(root='C:/Users/aleja/Datasets/CIFAKE/train', transform=transform)
total_train = len(total_data)
train = int(total_train * 0.8)
val = total_train - train

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = torch.utils.data.random_split(total_data, [train, val], generator=generator)

test_dataset = datasets.ImageFolder(root='C:/Users/aleja/Datasets/CIFAKE/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



