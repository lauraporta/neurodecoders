import torch
import torchvision
import torchvision.transforms as transforms

from neurodecoders.paths import ensure_dir, get_raw_datasets_path


class ImageDataset:
    def __init__(self):
        """Initialize common transform that's used across all methods."""
        # Basic transform without normalization (normalization will be calculated from training data)
        # Use native CIFAR-10 resolution (32x32) - similar scale to paper's 36x64
        self.base_transform = transforms.Compose(
            [
                # No resize - keep native resolution (32x32 for CIFAR-10, 28x28 for MNIST)
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        self.transform = self.base_transform
        self.mean = None
        self.std = None
    
    def calculate_normalization_stats(self, dataset_class, n_samples=5000):
        """
        Calculate mean and std from training data to match paper's approach.
        
        Args:
            dataset_class: The torchvision dataset class to use
            n_samples: Number of samples to use for computing statistics
        """
        root = get_raw_datasets_path()
        ensure_dir(root)
        dataset = dataset_class(
            root=root, train=True, download=True, transform=self.base_transform
        )
        
        # Use a subset for efficiency
        if n_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:n_samples].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=100, shuffle=False, num_workers=0
        )
        
        # Calculate mean and std across all images
        mean = 0.0
        std = 0.0
        n_pixels = 0
        
        for images, _ in loader:
            batch_pixels = images.numel()
            mean += images.sum()
            std += (images ** 2).sum()
            n_pixels += batch_pixels
        
        mean /= n_pixels
        std = torch.sqrt(std / n_pixels - mean ** 2)
        
        self.mean = mean.item()
        self.std = std.item()
        
        print(f"Calculated normalization stats: mean={self.mean:.4f}, std={self.std:.4f}")
        
        # Update transform with calculated normalization
        self.transform = transforms.Compose(
            [
                self.base_transform,
                transforms.Normalize(mean=[self.mean], std=[self.std]),
            ]
        )
        
        return self.mean, self.std
    
    def get_data(self, type: str, n_images: int, calculate_stats=True):
        """
        Get dataset with optional normalization stats calculation.
        
        Args:
            type: Dataset type ('cifar10' or 'mnist')
            n_images: Number of images to load
            calculate_stats: Whether to calculate and apply data-driven normalization
        """
        dataset_class = self._get_dataset_class(type)
        
        if calculate_stats and self.mean is None:
            self.calculate_normalization_stats(dataset_class)
        
        return self.load_dataset(dataset_class, n_images)
    
    def _get_dataset_class(self, type: str):
        """Get the dataset class for the given type."""
        if type == "cifar10":
            return torchvision.datasets.CIFAR10
        elif type == "mnist":
            return torchvision.datasets.MNIST
        else:
            raise ValueError(f"Invalid dataset type: {type}")

    def get_data_loader(self, type: str, n_images: int, batch_size: int = 100, calculate_stats=True):
        """
        Get a DataLoader for memory-efficient loading.
        
        Args:
            type: Dataset type ('cifar10' or 'mnist')
            n_images: Total number of images to use
            batch_size: Batch size for the DataLoader
            calculate_stats: Whether to calculate and apply data-driven normalization
            
        Returns:
            DataLoader and labels tensor
        """
        dataset_class = self._get_dataset_class(type)
        
        if calculate_stats and self.mean is None:
            self.calculate_normalization_stats(dataset_class)
        
        return self._create_loader(dataset_class, n_images, batch_size)
    
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
