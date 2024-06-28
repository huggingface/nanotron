if __name__ == "__main__":
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ]
    )

    trainset = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)
