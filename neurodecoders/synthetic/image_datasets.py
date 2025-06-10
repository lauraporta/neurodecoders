import torchvision
import torchvision.transforms as transforms
import torch



class ImageDataset:

    def get_data(self, type: str, n_images: int):
        if type == "cifar10":
            return self.load_dataset(torchvision.datasets.CIFAR10, n_images)
        elif type == "mnist":
            return self.load_dataset(torchvision.datasets.MNIST, n_images)
        else:
            raise ValueError(f"Invalid dataset type: {type}")

    def load_dataset(self, dataset_class, n_images: int):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=n_images, shuffle=True)
        images, _ = next(iter(loader))
        # normalize images to 0-1
        images = (images - images.min()) / (images.max() - images.min())
        return images[:n_images]
    

    