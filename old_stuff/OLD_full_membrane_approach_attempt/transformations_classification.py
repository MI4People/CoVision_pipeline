import numpy as np
from torchvision import transforms

class PreprocessFullMembrane:
    def __init__(self, w=224, h=224):

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.w, self.h = w, h

        # Common resizing transformation
        self.resize = transforms.Resize((self.h, self.w))
        # For data augmentation
        self.augment = None
        # Final common transformation
        self.final_transformation = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(self.mean, self.std)])
            
    def __call__(self, image):

        image = self.resize(image)
        if self.augment:
            image = self.augment(image)
        image = self.final_transformation(image)

        return image

class AugmentedFullMembrane(PreprocessFullMembrane):
    def __init__(self, w=248, h=248):
        super().__init__(w=w, h=h)
        self.augment = transforms.Compose([
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.GaussianBlur(kernel_size=(1, 9), sigma=(0.1, 5.))])