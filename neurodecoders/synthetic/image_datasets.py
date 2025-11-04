import torch
import torchvision
import torchvision.transforms as transforms

from neurodecoders.paths import ensure_dir, get_raw_datasets_path


class ImageDataset:
    def __init__(self):
        """Initialize common transform that's used across all methods."""
        self.transform = transforms.Compose(
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
    
    def get_data(self, type: str, n_images: int):
        if type == "cifar10":
            return self.load_dataset(torchvision.datasets.CIFAR10, n_images)
        elif type == "mnist":
            return self.load_dataset(torchvision.datasets.MNIST, n_images)
        else:
            raise ValueError(f"Invalid dataset type: {type}")

    def get_data_loader(self, type: str, n_images: int, batch_size: int = 100):
        """
        Get a DataLoader for memory-efficient loading.
        
        Args:
            type: Dataset type ('cifar10' or 'mnist')
            n_images: Total number of images to use
            batch_size: Batch size for the DataLoader
            
        Returns:
            DataLoader and labels tensor
        """
        if type == "cifar10":
            return self._create_loader(torchvision.datasets.CIFAR10, n_images, batch_size)
        elif type == "mnist":
            return self._create_loader(torchvision.datasets.MNIST, n_images, batch_size)
        else:
            raise ValueError(f"Invalid dataset type: {type}")
    
    def _create_loader(self, dataset_class, n_images: int, batch_size: int):
        """Create a DataLoader with the specified batch size."""
        root = get_raw_datasets_path()
        ensure_dir(root)
        dataset = dataset_class(
            root=root, train=True, download=True, transform=self.transform
        )
        
        # Use a subset if we don't need all images
        if n_images < len(dataset):
            # Use fixed seed for reproducibility across DataLoader creations
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:n_images].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Collect all labels efficiently in one pass
        all_labels = []
        for _, labels in loader:
            all_labels.append(labels)
            if len(torch.cat(all_labels)) >= n_images:
                break
        
        labels_tensor = torch.cat(all_labels)[:n_images]
        
        # Create a fresh loader to return (the previous one was exhausted)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        return loader, labels_tensor

    def load_dataset(self, dataset_class, n_images: int):
        root = get_raw_datasets_path()
        ensure_dir(root)
        dataset = dataset_class(
            root=root, train=True, download=True, transform=self.transform
        )
        
        # Use a subset if we don't need all images - sample without replacement
        if n_images < len(dataset):
            # Use fixed seed for reproducibility
            indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:n_images].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=n_images, shuffle=False
        )
        images, labels = next(iter(loader))
        # No need for additional normalization since ToTensor and Normalize
        # already give us [-1, 1]
        return images[:n_images], labels[:n_images]
