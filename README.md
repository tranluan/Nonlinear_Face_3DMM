# Nonlinear 3D Face Morphable Model
### [[Project page]](http://cvlab.cse.msu.edu/project-nonlinear-3dmm.html)   [[CVPR'18 Paper]](http://cvlab.cse.msu.edu/pdfs/Tran_Liu_CVPR2018.pdf) [[CVPR'19 Paper]](http://cvlab.cse.msu.edu/pdfs/Tran_Liu_Liu_CVPR2019.pdf)


![Teaser](./images/nonlinear-3dmm.jpg)

## Library requirements

* Tensorflow


## Data

Download following pre-processed training data (10GB) and unzip into ./data/300W_LP/

[Filelist](https://drive.google.com/open?id=1R80j6Y1JiNPzsucsMOGpoogKDiYg2ynP)
[Images](https://drive.google.com/open?id=1QkBiPAOA-a2buta--8atVVcKoAl5sj7O)
[Textures](https://drive.google.com/open?id=1oW8wTKkkw2VDVpCv9q8UjqG3mGQCHLQd)
[Masks](https://drive.google.com/open?id=1xTTtYYWIJlq8wYEl5BSQfjM-Vuw3jmwq)

Download following 3DMM definition and unzip into current folder (./)
[3DMM_definition.zip](https://drive.google.com/open?id=1-UJdQeFw0cf9u9gUHokNoheH0z3L1fEH)

## Compile the rendering layer - CUDA code
Please edit TF_newop/compile_op_v2_sz224.sh based on your TF version and whether you install TF with Anaconda (instruction in the file)

```bash
$ # Compile
$ cd TF_newop/
$ ./compile_op_v2_sz224.sh
$ # Run an example
$ python rendering_example.py
```
Currently the code is working but not optimal (i.e see line 139 of TF_newop/cuda_op_kernel_v2_sz224.cu.cc)
also the image size is hard-coded. Any contribution is welcome!


## Run the code

Note: In recent TF version, set --is_<some_thing> False (i.e --is_using_recon False) doesn't actually set it to False. In this case, you can just don't set it and use the default False value. Please print out those flags value to make sure.

Pretraining

```bash
python main_non_linear_3DMM.py --batch_size 128 --sample_size 128 --is_train True --learning_rate 0.001 --ouput_size  224 \
   --gf_dim 32 --df_dim 32 --dfc_dim 320 --gfc_dim 320 --z_dim 20 --c_dim 3 \
   --is_using_landmark True  --shape_loss l2 --tex_loss l1 \
   --is_using_recon False --is_using_frecon False --is_partbase_albedo False --is_using_symetry True  \
   --is_albedo_supervision False --is_batchwise_white_shading True --is_const_albedo True --is_const_local_albedo False --is_smoothness True 
  --gpu 0,1,2,3
```


Finetunning
Manually reduce the m_loss, shape_loss weight by 10 times

```bash
python main_non_linear_3DMM.py --batch_size 64 --sample_size 64 --is_train True --learning_rate 0.001 --ouput_size  224 \
   --gf_dim 32 --df_dim 32 --dfc_dim 320 --gfc_dim 320 --z_dim 20 --c_dim 3 \
   --is_using_landmark True  --shape_loss l2 --tex_loss l1 \
   --is_using_recon True --is_using_frecon True --is_partbase_albedo False --is_using_symetry True  \
   --is_albedo_supervision False --is_batchwise_white_shading True --is_const_albedo True --is_const_local_albedo True --is_smoothness True 
  --gpu 0,1,2,3 \
```

## Pretrain model

This is the [pretrained model](https://www.cse.msu.edu/computervision/evaluation_code.zip) of CVPR'19 paper. Input images are 256 x 256.


## Citation

If you find this work useful, please cite our papers with the following bibtex:
```latex
@inproceedings{ tran2019towards, 
  author = { Luan Tran and Feng Liu and Xiaoming Liu },
  title = { Towards High-fidelity Nonlinear 3D Face Morphable Model },
  booktitle = { In Proceeding of IEEE Computer Vision and Pattern Recognition },
  address = { Long Beach, CA },
  month = { June },
  year = { 2019 },
}
```

```latex
@article{ tran2018on, 
  author = { Luan Tran and Xiaoming Liu },
  title = { On Learning 3D Face Morphable Model from In-the-wild Images },
  journal = { IEEE Transactions on Pattern Analysis and Machine Intelligence },
  month = { July },
  year = { 2019 },
}
```


```latex
@inproceedings{ tran2018nonlinear, 
  author = { Luan Tran and Xiaoming Liu },
  title = { Nonlinear 3D Face Morphable Model },
  booktitle = { IEEE Computer Vision and Pattern Recognition (CVPR) },
  address = { Salt Lake City, UT },
  month = { June },
  year = { 2018 },
}
```

## Contacts

If you have any questions, feel free to drop an email to _tranluan@msu.edu_.
