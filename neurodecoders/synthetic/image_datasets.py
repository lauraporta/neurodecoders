import torch
import torchvision
import torchvision.transforms as transforms

from neurodecoders.paths import ensure_dir, get_raw_datasets_path


class ImageDataset:
    # check that it is not RGB
    def get_data(self, type: str, n_images: int):
        if type == "cifar10":
            return self.load_dataset(torchvision.datasets.CIFAR10, n_images)
        elif type == "mnist":
            return self.load_dataset(torchvision.datasets.MNIST, n_images)
        else:
            raise ValueError(f"Invalid dataset type: {type}")

    def load_dataset(self, dataset_class, n_images: int):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                    # NB: this is not the normalization
                    # suggested for RGB images
                ),  # Normalize to [-1, 1] with mean=0.5
            ]
        )
        root = get_raw_datasets_path()
        ensure_dir(root)
        dataset = dataset_class(
            root=root, train=True, download=True, transform=transform
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=n_images, shuffle=True
        )
        images, labels = next(iter(loader))
        # No need for additional normalization since ToTensor and Normalize
        # already give us [-1, 1]
        return images[:n_images], labels[:n_images]
