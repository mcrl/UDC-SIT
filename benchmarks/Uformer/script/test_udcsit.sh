python3 src/test_udcsit.py --arch Uformer_T --H 1792 --W 1280 --batch_size_val 8 --gpu '0,1,2,3' \
    --test_dir /data/s0/udc/dataset/UDC_SIT/npy/test1 --patch_size 1280 \
    --val_ps 1280 --val_dir /data/s0/udc/dataset/UDC_SIT/npy/validation1 --env _0518 --dataset_name UDC_SIT \
    --mode motiondeblur --checkpoint 50 --dataset UDC_SIT --warmup --dd_in 4 

