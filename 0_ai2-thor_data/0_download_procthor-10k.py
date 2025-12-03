import os
import prior
import shutil
import gzip

prior.DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
dataset = prior.load_dataset("procthor-10k")

downloaded_base_dir = os.path.join(prior.DATASET_DIR, "allenai", "procthor-10k")

sha_dirs = [d for d in os.listdir(downloaded_base_dir) if os.path.isdir(os.path.join(downloaded_base_dir, d))]
if sha_dirs:
    sha_path = max((os.path.join(downloaded_base_dir, d) for d in sha_dirs), key=lambda p: os.stat(p).st_mtime)

    final_target_dir = os.path.join(prior.DATASET_DIR, "procthor-10k")
    os.makedirs(final_target_dir, exist_ok=True)

    # Move and decompress .gz files
    for filename in ["test.jsonl.gz", "train.jsonl.gz", "val.jsonl.gz"]:
        src_path = os.path.join(sha_path, filename)
        dest_path_gz = os.path.join(final_target_dir, filename)
        dest_path_jsonl = os.path.join(final_target_dir, filename.replace(".gz", ""))

        if os.path.exists(src_path):
            # Move the .gz file
            shutil.move(src_path, dest_path_gz)
            # Decompress the file
            with gzip.open(dest_path_gz, 'rb') as f_in:
                with open(dest_path_jsonl, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(dest_path_gz) # Remove the .gz file after decompression

# Clean up the allenai directory
allenai_dir = os.path.join(prior.DATASET_DIR, "allenai")
if os.path.exists(allenai_dir):
    shutil.rmtree(allenai_dir)

