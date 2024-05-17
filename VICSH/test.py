from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageFolder

train_transforms = transforms.Compose(
        [transforms.RandomCrop(size=(256, 256)), transforms.ToTensor()]
    )
test_transforms = transforms.Compose(
        [transforms.CenterCrop(size=(256, 256)), transforms.ToTensor()]
    )

trainset_dir = '/media/D/dataset/openimages/train_1'
train_dataset = ImageFolder(trainset_dir, transform=train_transforms)
train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

testset_dir = '/media/D/dataset/kodak_test'
test_dataset = ImageFolder(testset_dir, transform=test_transforms)
test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

print(len(train_dataloader))
print(len(test_dataloader))