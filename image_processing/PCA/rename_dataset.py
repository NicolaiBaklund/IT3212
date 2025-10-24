import os

base_dir = "data/celeb_faces"


def rename_dataset(base_dir: str = base_dir):
    """
    Renames folders to lowercase with underscores instead of spaces.
    Renames images in each folder to img00, img01, ..., preserving extensions.
    """
    for folder in os.listdir(base_dir):
        old_path = os.path.join(base_dir, folder)
        if not os.path.isdir(old_path):
            continue  # skip files if any

        # --- Rename folder ---
        new_folder_name = folder.lower().replace(" ", "_")
        new_path = os.path.join(base_dir, new_folder_name)

        # Only rename if needed
        if new_path != old_path:
            os.rename(old_path, new_path)
            print(f"Renamed folder: {folder} -> {new_folder_name}")
        else:
            print(f"Folder name already OK: {folder}")

        # --- Rename images ---
        images = sorted(os.listdir(new_path))
        for i, filename in enumerate(images):
            old_file = os.path.join(new_path, filename)

            # Get file extension (handles .jpg/.jpeg/.png etc.)
            ext = os.path.splitext(filename)[1].lower()
            new_filename = f"img{i:02d}{ext}"
            new_file = os.path.join(new_path, new_filename)

            if os.path.exists(new_file):
                print(f"Warning: {new_filename} already exists in {new_folder_name}. Skipping rename of {filename}.")
                continue
            os.rename(old_file, new_file)

        print(f"Renamed {len(images)} images in {new_folder_name}")
