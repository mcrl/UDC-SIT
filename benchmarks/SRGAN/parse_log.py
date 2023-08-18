filepath = "logs/Feng_final_test.log"
with open(filepath, "r") as f:
    lines = f.readlines()

with open("Feng.csv", "w") as f:
    f.write("image,PSNR,SSIM\n")
    for line in lines:
        if "Average SSIM" in line:
            continue
        line = line.strip()
        tokens = line.split(",")

        img_name = tokens[0].strip().replace("Img name: ", "")
        psnr = tokens[1].strip().replace("PSNR: ", "")
        ssim = tokens[2].strip().replace("SSIM: ", "")
        print(img_name, psnr, ssim)
        f.write(f"{img_name},{psnr},{ssim}\n")
