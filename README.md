# R2-V2

Code for running R2-V2 models for artery/vein segmentation in retinal fundus images.


## Available models

There are two models available, `av` and `bv`. Both of them perform vessel segmentation and artery/vein classification. However, they have different characteristics:

+ `av`: This model is mostly focused on the correct classification of arteries and veins, with a very high vessel sensitivity.
+ `bv`: This model is more balanced, and performs particularly well for vessel segmentation.

The weights and the configuration files of the models are available at the **Releases** section.


## Usage

The implementation is based on Python 3.12.8, PyTorch 2.8, and CUDA 12.8.
It also uses other libraries, such as scikit-image and NumPy, for image processing and transformations.

Thus, for running the code, it is first necessary to set up the environment.


<details>
<summary><b>Installing Python 3.12.8 using pyenv</b></summary>

### Python 3.12.8 (`pyenv`)

> **📌 IMPORTANT**: The following steps are only necessary if you want to install Python 3.12.8 using `pyenv`.

Install `pyenv`.
```sh
curl https://pyenv.run | bash
```

Install `clang`. _E.g._:
```sh
sudo dnf install clang
```

Install Python version 3.12.8.
```sh
CC=clang pyenv install -v 3.12.8
```

Create and activate Python environment.
```sh
~/.pyenv/versions/3.12.8/bin/python3 -m venv venv/
source venv/bin/activate  # bash
. venv/bin/activate.fish  # fish
```

Update `pip` if necessary.

```sh
pip install --upgrade pip
```

</details>



### Requirements

Create and activate Python environment.
```sh
python -m venv venv/
source venv/bin/activate  # bash
. venv/bin/activate.fish  # fish
```

Install requirements using `requirements.txt`.

```sh
pip3 install -r requirements.txt
```


### Running the code

To run the code, use the `infer.py` script. Please check the available options using the `-h` or `--help` flag.

The models need preprocessed images in order to work well. If no directory with preprocessed images is indicated, the inference code will preprocess the image on the fly. However, it is also possible to preprocess all the images before running inference using `preprocessing.py`. Again, please check the available options using the `-h` or `--help` flag.

