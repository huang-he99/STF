unet input 

x=torch.cat( (coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02), dim=1)
