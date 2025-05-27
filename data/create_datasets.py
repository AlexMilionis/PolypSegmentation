# from N numbers (1-N) select 10% of them randomly and return them as a list
import random
random.seed(42)  # For reproducibility
import os

def select_random_numbers(n, prc=0.1):
    samples_list = list(range(1, n+1))
    val_list = random.sample(samples_list, int(n*prc))
    val_list.sort()
    samples_list = [sample for sample in samples_list if sample not in val_list]
    test_list = random.sample(samples_list, int(n*prc))
    test_list.sort()
    train_list = [sample for sample in samples_list if sample not in test_list]
    return train_list, val_list, test_list


# function that reads folder contents and returns the names of the files in a list
def fetch_val_test_images_masks(folder_path):
    image_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    train_list, val_list, test_list = select_random_numbers(len(image_names), 0.1)
    train_images = [image_names[i-1] for i in train_list]
    val_images = [image_names[i-1] for i in val_list]
    test_images = [image_names[i-1] for i in test_list]

    train_masks = [image_names[i-1].replace('.jpg', '_mask.jpg') for i in train_list]
    val_masks = [image_names[i-1].replace('.jpg', '_mask.jpg') for i in val_list]
    test_masks = [image_names[i-1].replace('.jpg', '_mask.jpg') for i in test_list]
    return train_images, val_images, test_images, train_masks, val_masks, test_masks


# function copies val_images, test_images, val_masks, test_masks to new folders
def cut_images_masks(train_images, val_images, test_images, train_masks, val_masks, test_masks, center):

    for image in train_images:
        src_folder = f"data/data_C{center}/images_C{center}"
        src_image_path = os.path.join(src_folder, image)
        dest_image_path = os.path.join("data", "train", "images", image)
        if os.path.exists(src_image_path):
            os.rename(src_image_path, dest_image_path)

    for image in val_images:
        src_folder = f"data/data_C{center}/images_C{center}"
        src_image_path = os.path.join(src_folder, image)
        dest_image_path = os.path.join("data", "val", "images", image)
        if os.path.exists(src_image_path):
            os.rename(src_image_path, dest_image_path)

    for image in test_images:
        src_folder = f"data/data_C{center}/images_C{center}"
        src_image_path = os.path.join(src_folder, image)
        # dest_image_path = os.path.join(dest_folder, 'test', image)
        dest_image_path = os.path.join("data", "test", "images", image)
        if os.path.exists(src_image_path):
            os.rename(src_image_path, dest_image_path)

    for mask in train_masks:
        src_folder = f"data/data_C{center}/masks_C{center}"
        src_mask_path = os.path.join(src_folder, mask)
        # dest_mask_path = os.path.join(dest_folder, 'val', mask)
        dest_mask_path = os.path.join("data", "train", "masks", mask)
        if os.path.exists(src_mask_path):
            os.rename(src_mask_path, dest_mask_path)

    for mask in val_masks:
        src_folder = f"data/data_C{center}/masks_C{center}"
        src_mask_path = os.path.join(src_folder, mask)
        # dest_mask_path = os.path.join(dest_folder, 'val', mask)
        dest_mask_path = os.path.join("data", "val", "masks", mask)
        if os.path.exists(src_mask_path):
            os.rename(src_mask_path, dest_mask_path)

    for mask in test_masks:
        src_folder = f"data/data_C{center}/masks_C{center}"
        src_mask_path = os.path.join(src_folder, mask)
        # dest_mask_path = os.path.join(dest_folder, 'test', mask)
        dest_mask_path = os.path.join("data", "test", "masks", mask)
        if os.path.exists(src_mask_path):
            os.rename(src_mask_path, dest_mask_path)


for center in range(2, 7):
    train_images, val_images, test_images, train_masks, val_masks, test_masks = fetch_val_test_images_masks(f"data/data_C{center}/images_C{center}")
    cut_images_masks(train_images, val_images, test_images, train_masks, val_masks, test_masks, center=center)



    