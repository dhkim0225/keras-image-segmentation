# Keras Image Segmentation

Semantic Segmentation easy code for keras users.

We use [Cityscape dataset](https://www.cityscapes-dataset.com/) for training various models.

Use pretrained VGG16 weight! You can [download weights](
'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
offered by keras.

### Tested Env
- python 2 & 3
- tensorflow 1.5
- keras 2.1.4

### Simple Tutorial
First, you have to make .h5 file with data!
```bash
python make_h5.py --path "/downloaded/leftImg8bit/path/" --gtpath "/downloaded/gtFine/path/"
```

Second, Train your model!

Finally, test your model!

### Todo
- [x] FCN
- [x] Unet
- [ ] PSPnet
- [ ] DeepLab_v1
- [ ] DeepLab_v2
- [ ] DeepLab_v3

### Contact us!
Anthony Kim: artit.anthony@gmail.com

TaeKang Woo: wtk1101@gmail.com