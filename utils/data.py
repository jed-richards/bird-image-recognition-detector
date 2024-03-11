import os

import pandas as pd

data = {
    "dataset": [],  # train, test, or valid
    "filepath": [],  # path to image
    "label": [],  # bird label
}

data_root = "data"
exclude_dirs = ["data", "data/train", "data/test", "data/valid"]
for root, dirs, files in os.walk(data_root):
    if root in exclude_dirs:
        continue

    # Generate dataset, filepath, and label lists for each subdirectory
    paths = [os.path.join(root, file) for file in files]
    _, dataset, label = root.split("/")  # get dataset and label from "data/train/LABEL"
    num_images = len(paths)
    labels_list = [label] * num_images
    dataset_list = [dataset] * num_images

    # Append lists to data dictionary
    data["dataset"].extend(dataset_list)
    data["filepath"].extend(paths)
    data["label"].extend(labels_list)

df = pd.DataFrame(data=data)
df = df.sort_values(by=["dataset", "label"])

# Extract unique species and assign class IDs
unique_species = df["label"].unique()
class_ids = pd.factorize(unique_species)[0]

# Create a dictionary mapping species to class IDs
species_to_classid = dict(zip(unique_species, class_ids))

# Add a new 'class_id' column to the DataFrame
df["class_id"] = df["label"].map(species_to_classid)

df.to_csv("data/bird_df.csv", index=False)

# PLOT SOME IMAGES TO SEE HOW THEY LOOK
# test_df = df[df['dataset'] == 'test']
# train_df = df[df['dataset'] == 'train']
# valid_df = df[df['dataset'] == 'valid']
#
# plt.figure(figsize=(15,12))
# for i, row in valid_df.sample(n=16).reset_index().iterrows():
#    plt.subplot(4,4,i+1)
#    image_path = row['filepath']
#    image = Image.open(image_path)
#    print(np.array(image).shape)
#    plt.imshow(image)
#    plt.title(row["label"])
#    plt.axis('off')
# plt.show()
#
# print(len(df))
# print(df.shape[0])
