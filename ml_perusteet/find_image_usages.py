import glob
import re
import argparse
from pathlib import Path

def list_images(input_dir:Path) -> list[Path]:
    image_files = []
    for ext in ['*.png', '*.jpg', '*.svg']:
        image_files += Path(input_dir).glob(ext)
    return image_files

def find_image_mentions(image_files:list[Path], target_dir:Path):
    markdown_files = target_dir.rglob('*.md')
    image_mentions = {}
    for md_file in markdown_files:
        with open(md_file, 'r') as f:
            content = f.read()
            for image_file in image_files:
                if image_file.name in content:
                    image_mentions[image_file.name] = md_file
    return image_mentions

def main():
    parser = argparse.ArgumentParser(description='Find images mentioned in Markdown files.')
    parser.add_argument('-i', '--imgs', default="docs/images", type=str, help='Input directory containing images')
    parser.add_argument('-m', '--mds', default="docs", type=str, help='Markdown files directory')
    args = parser.parse_args()

    image_dir = Path(args.imgs)
    markdown_dir = Path(args.mds)

    print(f"[INFO] Scanning '{image_dir}' for images")
    image_files: list[Path] = list_images(image_dir)
    print(f"[INFO] Finding matches in '{markdown_dir}' markdown files")
    image_mentions = find_image_mentions(image_files, markdown_dir)

    # Check for unnecessary image files
    unnecessary_images = set([x.name for x in image_files]) - set(image_mentions.keys())
    if unnecessary_images:
        print(f"\n[WARNING] {len(unnecessary_images)} unnecessary image files found:")
        for image in unnecessary_images:
            print(image)

if __name__ == "__main__":
    main()
