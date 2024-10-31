import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder


class FoodDataset(ImageFolder):
    def __init__(self, root, limit=None):
        super(FoodDataset, self).__init__(
            root,
            transform=transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            loader=lambda img: Image.open(img).convert("RGB"),
        )
        if limit:
            self.samples = self.samples[:limit]
            self.targets = self.targets[:limit]
            self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        class_name = self.classes[target]
        return {"img": image, "label": target, "food_class_name": class_name}
