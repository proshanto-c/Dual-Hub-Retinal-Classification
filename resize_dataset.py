import os
import argparse
from pathlib import Path
from PIL import Image
import multiprocessing
from tqdm import tqdm

def resize_image(args):
    input_path, output_path, size = args
    if os.path.exists(output_path):
        return True
        
    try:
        with Image.open(input_path) as img:
            # Convert to RGB to ensure consistency
            img = img.convert("RGB")
            
            # Using LANCZOS (formerly ANTIALIAS) for high-quality downsampling
            resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # Save the new image at high quality
            resized_img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pre-resize the BRSET dataset for extremely fast DataLoader I/O.")
    parser.add_argument("--input-dir", type=str, default="data/BRSET/fundus_photos", help="Path to original high-res photos.")
    parser.add_argument("--output-dir", type=str, default="data/BRSET/fundus_photos_512", help="Where to save resized photos.")
    parser.add_argument("--size", type=int, default=512, help="Target image size (e.g. 512 for 512x512).")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of CPU threads to use.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found!")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all images
    print("Scanning directory for images...")
    image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    print(f"Found {len(image_paths)} images.")
    
    # Prepare arguments for multiprocessing pool
    tasks = [(p, output_dir / p.name, args.size) for p in image_paths]
    
    print(f"Starting highly-parallel resize using {args.workers} CPU cores...")
    
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(resize_image, tasks), total=len(tasks), desc="Resizing"))
        
    successful = sum(1 for r in results if r)
    print(f"\nDone! Successfully resized {successful}/{len(tasks)} images.")
    print(f"New dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()
