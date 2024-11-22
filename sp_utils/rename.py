import os
import sys

def rename_folders(directory, prefix):
    # Get a sorted list of all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    folders.sort()

    # Rename folders sequentially with the given prefix
    for index, folder in enumerate(folders, start=1):
        new_folder_name = f"{prefix}_{index:03}"  # Format index with leading zeros, e.g., prefix_001
        old_path = os.path.join(directory, folder)
        new_path = os.path.join(directory, new_folder_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {folder} -> {new_folder_name}")

if __name__ == "__main__":
    # Check if the directory and prefix arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python rename_script.py <directory_path> --prefix <prefix_value>")
        sys.exit(1)

    # Get the directory path from the command-line argument
    target_directory = sys.argv[1]

    # Check if '--prefix' argument is provided and get its value
    if "--prefix" not in sys.argv:
        print("Error: Please provide the --prefix argument.")
        sys.exit(1)

    prefix_index = sys.argv.index("--prefix")
    if prefix_index + 1 >= len(sys.argv):
        print("Error: Missing value for --prefix.")
        sys.exit(1)

    prefix = sys.argv[prefix_index + 1]

    # Check if the provided path is valid
    if not os.path.isdir(target_directory):
        print(f"Error: {target_directory} is not a valid directory")
        sys.exit(1)

    # Call the rename function
    rename_folders(target_directory, prefix)
