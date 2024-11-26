from medmnist import BreastMNIST
from medmnist import BloodMNIST



# download BreastMNIST
BreastMNIST(root='./Datasets', split='train', download=True)
BreastMNIST(root='./Datasets', split='val', download=True)
BreastMNIST(root='./Datasets', split='test', download=True)

# download BloodMNIST
BloodMNIST(root='./Datasets', split='train', download=True)
BloodMNIST(root='./Datasets', split='val', download=True)
BloodMNIST(root='./Datasets', split='test', download=True)

