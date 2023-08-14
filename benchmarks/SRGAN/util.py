import rawpy as rp
import imageio
import torch


def save_4ch_npy_to_img(
    img_ts: torch.Tensor,
    img_path: str,
    dng_info="/home/n5/chanwoo/background.dng",
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


def save_3ch_npy_to_img(img_ts: torch.Tensor, img_path: str):
    pass


def setup_img_save_function(channels: int):
    global img_save_function
    if channels == 3:
        print("Setting up 3 channel image save function")
        img_save_function = save_3ch_npy_to_img
    elif channels == 4:
        print("Setting up 4 channel image save function")
        img_save_function = save_4ch_npy_to_img
    else:
        raise ValueError("Channels must be 3 or 4")
