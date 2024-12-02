"""
Deletes the image and mask directories specified in Constants.

The script removes the directories completely, including all their contents.
A confirmation prompt is displayed to avoid accidental deletion.

Raises:
    FileNotFoundError: If any of the directories do not exist.
"""


import shutil
import os
from constants import Constants


def delete_data():
    try:
        # Confirmation prompt
        confirm = input(f"Are you sure you want to delete the following directories?\n"
                        f"1. {Constants.IMAGE_DIR}\n"
                        f"2. {Constants.MASK_DIR}\n"
                        f"Type 'yes' to confirm: ").strip().lower()
        if confirm != 'yes':
            print("Deletion aborted.")
            return

        # Delete image directory
        if os.path.exists(Constants.IMAGE_DIR):
            shutil.rmtree(Constants.IMAGE_DIR)
            print(f"Deleted: {Constants.IMAGE_DIR}")
        else:
            print(f"Directory does not exist: {Constants.IMAGE_DIR}")

        # Delete mask directory
        if os.path.exists(Constants.MASK_DIR):
            shutil.rmtree(Constants.MASK_DIR)
            print(f"Deleted: {Constants.MASK_DIR}")
        else:
            print(f"Directory does not exist: {Constants.MASK_DIR}")

    except Exception as e:
        print(f"An error occurred while deleting directories: {e}")


if __name__ == '__main__':
    delete_data()
