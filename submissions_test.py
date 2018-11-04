def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255
           for color in colors]
    return np.stack(img, axis=-1)


    def get_data(sz,bs):
        #data augmentation
        aug_tfms = [RandomRotate(30, tfm_y=TfmType.NO),
                    RandomDihedral(tfm_y=TfmType.NO),
                    RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
        #mean and std in of each channel in the train set
        stats = A([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])
        tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                    aug_tfms=aug_tfms)
        ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN),
                    (val_n,TRAIN), tfms, test=(test_names,TEST))
        md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
        return md
