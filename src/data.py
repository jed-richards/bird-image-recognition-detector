import os
import pandas as pd

data = {
    "dataset": [],   # train, test, or valid
    "filepath": [],  # path to image
    "label": [],     # bird label
}

data_root = "data"
exclude_dirs = ["data", "data/train", "data/test", "data/valid"]
for root, dirs, files in os.walk(data_root):
    if root in exclude_dirs:
        continue

    # Generate dataset, filepath, and label lists for each subdirectory
    paths = [os.path.join(root, file) for file in files]
    _, dataset, label = root.split("/") # get dataset and label from "data/train/LABEL"
    num_images = len(paths)
    labels_list = [label]*num_images
    dataset_list = [dataset]*num_images

    # Append lists to data dictionary
    data["dataset"].extend(dataset_list)
    data["filepath"].extend(paths)
    data["label"].extend(labels_list)

df = pd.DataFrame(data=data)
df.to_csv("bird_df.csv", index=False)
print(df)

#df = pd.read_csv("bird_df.csv")
#print(df)
