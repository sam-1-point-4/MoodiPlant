import os


def rename_folders(directory):
    for folder_name in os.listdir(directory):
        new_folder_name = folder_name.replace(' ', '_')

        old_path = os.path.join(directory, folder_name)
        new_path = os.path.join(directory, new_folder_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {folder_name} -> {new_folder_name}")


dataset_dir = "C:/Users/ezgit/Documents/moodiplant/data"

rename_folders(os.path.join(dataset_dir, "test"))
rename_folders(os.path.join(dataset_dir, "train"))
