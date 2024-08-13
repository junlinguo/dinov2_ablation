# visualization

## features extraction visualization 
```python

"""
analyze features extracted by backbone 

virtual env: sarl_env_xformers
"""
import torch
import timm
import albumentations as A
from Custom_dataset import CustomImageDataset
from torch.utils.data import DataLoader

from tqdm import tqdm 

# data 
data_file = '/home/guoj5/Desktop/correct_labels/train_unbalanced.csv'
data_augmentation_transform = A.Compose([
        # resize to be product of 14
        A.Resize(224, 224),
        A.Flip(),
        A.RandomRotate90(),
    ])

train_dataset = CustomImageDataset(data_file= data_file,
                                   transform=data_augmentation_transform,
                                   edge_channel=False)
train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False
    )

# vit large - imagenet 
backbone = 'vit_large_patch16_224'


# model 
model = timm.create_model(model_name=backbone, pretrained=True, in_chans=8)

# hook 
def get_features_hook(module, input, output):
    global features_before_head
    features_before_head.append(output)

# Attach the hook to the layer before the linear head
hook = model.fc_norm.register_forward_hook(get_features_hook)


# Initialize a list to store features and labels
features_before_head = []
all_labels = []

# Iterate over the dataloader
model.eval()
with torch.no_grad():
    for batch in tqdm(train_loader):
        images, batch_labels = batch
        outputs = model(images)
        all_labels.extend(batch_labels)
        # The hook automatically saves the features into the features_before_head list

# After all batches are processed, remove the hook
hook.remove()

# Convert the list of features to a tensor for further analysis
features_before_head = torch.cat(features_before_head, dim=0)  # Shape: [num_images, hidden_dim]
all_labels = torch.tensor(all_labels)  # Convert labels to a tensor

print()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Alternatively, use t-SNE
tsne = TSNE(n_components=2, perplexity=30)
features_2d = tsne.fit_transform(features_before_head.numpy())
# Plot the 2D representation
plt.figure(figsize=(8, 8))
unique_labels = set(all_labels.numpy())
for label in unique_labels:
    indices = [i for i, lbl in enumerate(all_labels) if lbl == label]
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label)
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Class Token Distribution")

plt.savefig('/home/guoj5/Desktop/prototype/visualization/class_token_distribution.png')
plt.show()


```
