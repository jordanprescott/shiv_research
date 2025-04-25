import os
from PIL import Image

# Input/output paths
input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/1/photo"
output_gif = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/1/photo.gif"

def make_gif(input_dir, output_path):
    images = []
    # Sort filenames numerically by stripping extension and converting to int
    for filename in sorted(
        os.listdir(input_dir),
        key=lambda x: int(os.path.splitext(x)[0])
    ):
        if filename.endswith(('.png', '.jpg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert("RGBA")
            images.append(img)

    if not images:
        print("No images found in", input_dir)
        return

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=100
    )
    print(f"GIF saved to {output_path}")

# Run
make_gif(input_dir, output_gif)
