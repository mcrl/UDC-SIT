import rawpy as rp
import imageio
import torch
import numpy as np
from PIL import Image


def save_4ch_npy_to_img(
    img_ts: torch.Tensor,
    img_path: str,
    dng_info="../../background.dng",
):
    if dng_info is None:
        raise RuntimeError(
            "DNG information for saving 4 channeled npy file not provided"
        )
    # npy file in hwc manner to cwh manner
    data = rp.imread(dng_info)
    ts = img_ts.permute(0, 1, 3, 2)
    ts = torch.clamp(ts, 0, 1) * 1023
    ts = ts.cpu().numpy()

    GR = data.raw_image[0::2, 0::2]
    R = data.raw_image[0::2, 1::2]
    B = data.raw_image[1::2, 0::2]
    GB = data.raw_image[1::2, 1::2]
    GB[:, :] = 0
    B[:, :] = 0
    R[:, :] = 0
    GR[:, :] = 0

    w, h = (1280, 1792)

    GR[:w, :h] = ts[0][0][:w][:h]
    R[:w, :h] = ts[0][1][:w][:h]
    B[:w, :h] = ts[0][2][:w][:h]
    GB[:w, :h] = ts[0][3][:w][:h]
    start = (0, 464)  # (448 , 0) # 1792 1280 ->   3584, 2560
    end = (3584, 3024)
    newData = data.postprocess()
    output = newData[start[0] : end[0], start[1] : end[1]]
    # output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    imageio.imsave(img_path, output)


def save_3ch_npy_to_img(tensor: torch.Tensor, img_path: str):
    # squeeze
    tensor = tensor.squeeze(0)

    # Convert PyTorch tensor to NumPy array
    numpy_array = tensor.cpu().numpy()

    # Normalize and convert to uint8 type
    minimum = numpy_array.min()
    maximum = numpy_array.max()
    delta = maximum - minimum
    if delta < 1e-6:
        delta = 1
    normalized_array = (numpy_array - minimum) / delta

    # Convert to uint8 type. Input: 3 channel, output: 3 channel
    uint8_array = (normalized_array * 255.0).astype(np.uint8)
    uint8_array = np.transpose(uint8_array, (1, 2, 0))

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(uint8_array, mode="RGB")

    # Save PIL Image as PNG
    pil_image.save(img_path)


def setup_img_save_function(channels: int):
    global img_save_function
    if channels == 3:
        print("Setting up 3 channel image save function")
        return save_3ch_npy_to_img
    elif channels == 4:
        print("Setting up 4 channel image save function")
        return save_4ch_npy_to_img
    else:
        raise ValueError("Channels must be 3 or 4")
