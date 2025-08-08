import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil
import time
from romatch import roma_indoor
from typing import Tuple, Optional
from pathlib import Path


class RoMaImageMatcher:
    def __init__(self, coarse_res: int = 420, upsample_res: Tuple[int, int] = (648, 864)):
        """
        Initialize the RoMa image matcher.

        Args:
            coarse_res: Coarse resolution for the model
            upsample_res: Tuple of (height, width) for upsampling resolution
        """
        self.device = self._get_device()
        self.roma_model = roma_indoor(
            device=self.device,
            coarse_res=coarse_res,
            upsample_res=upsample_res
        )
        self.H, self.W = self.roma_model.get_output_resolution()

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def load_image(self, im_path: str) -> Image.Image:
        image = Image.open(im_path).resize((self.W, self.H))
        return image
    
    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        x = (torch.tensor(np.array(image)) / 255).to(self.device).permute(2, 0, 1)
        return x

    def match_images(self, image1: torch.Tensor, image2: torch.Tensor, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, float]:
        start_time = time.time()
        warp, certainty = self.roma_model.match(image1, image2, device=self.device)
        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Matching completed in {elapsed_time:.3f} seconds")

        return warp, certainty, elapsed_time

    def create_warp_visualization(self, x1: torch.Tensor,
                                        warp: torch.Tensor,
                                        certainty: torch.Tensor) -> torch.Tensor:
        # Extract warp for image 1 -> image 2 (assuming warp has shape: (H, 2*W, 2))
        # According to your original code, the right half of warp corresponds to image 1 -> image 2 flow:
        # warp[:, W:, :2]  (shape: H x W x 2)
        # We'll use that to warp x1.

        H, W = x1.shape[1], x1.shape[2]
        warp_1_to_2 = warp[:, W:, :2]  # H x W x 2

        # Add batch dimension and permute to grid_sample format (N, H, W, 2)
        grid = warp_1_to_2.unsqueeze(0)  # (1, H, W, 2)

        # Warp x1 to image 2 frame
        warped_x1 = F.grid_sample(
            x1.unsqueeze(0), grid,
            mode="bilinear", align_corners=False
        )[0]  # Remove batch dimension

        # Blend with white background based on certainty (right half corresponds to image 1 -> image 2)
        certainty_1_to_2 = certainty[:, W:]  # H x W
        white_bg = torch.ones_like(warped_x1)

        # Certainty needs to be broadcast to C x H x W
        certainty_exp = certainty_1_to_2.unsqueeze(0)  # 1 x H x W

        vis_im = certainty_exp * warped_x1 + (1 - certainty_exp) * white_bg

        return vis_im

    def save_visualization(self, vis_tensor: torch.Tensor, save_path: str):
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(vis_tensor, unnormalize=False).save(save_path)


import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

def main():
    width = 1000
    height = 750

    model = RoMaImageMatcher(
        coarse_res=532,
        upsample_res=(height, width)
    )

    try:
        while True:
            # Receive the raw data
            data = socket.recv()

            # Convert to NumPy array
            arr = np.frombuffer(data, dtype=np.uint8)

            # Check that the size matches
            assert arr.size == 2 * 3 * height * width, f"Unexpected data size: {arr.size}"

            # Reshape to (2, height, width, 3)
            arr = arr.reshape(2, height, width, 3)

            # Split into image1 and image2
            image1 = Image.fromarray(arr[0], mode='RGB')
            image2 = Image.fromarray(arr[1], mode='RGB')

            warp, certainty, _ = model.match_images(image1, image2)

            warp = warp.cpu()
            certainty = certainty.cpu()

            data1 = warp.numpy().tobytes()
            data2 = certainty.numpy().tobytes()

            socket.send_multipart([data1, data2])

            # x1 = model.image_to_tensor(image1)
            # vis = model.create_warp_visualization(x1, warp, certainty)
            # model.save_visualization(vis, "vis.jpg")
    except KeyboardInterrupt:
        print("\n[Server] Shutting down via Ctrl+C")


if __name__ == "__main__":
    main()
