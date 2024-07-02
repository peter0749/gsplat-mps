# gsplat-mps

This is a fork of [gsplat](https://github.com/nerfstudio-project/gsplat) version 0.1.3 ported to Apple MPS (Metal Performance Shaders), thanks to [OpenSplat's Metal implementation](https://github.com/pierotofy/OpenSplat/tree/main/rasterizer/gsplat-metal). It is not thoroughly tested, but I can confirm that (at time of writing) the `examples/image_fitting.py` script runs correctly on my device. Below is the quick start I used to get this module up and running on my device - please see the original repository's readme for more information.

_(Tested with Python 3.10.14)_.

```sh
git clone --recursive https://github.com/iffyloop/gsplat-mps.git
cd gsplat-mps
python -m virtualenv venv
source venv/bin/activate
pip install torch torchvision
pip install -e ".[dev]"
pip install -r examples/requirements.txt
python examples/simple_trainer.py
```

## License

`gsplat-mps` is licensed under AGPLv3 terms due to the Metal implementation derived from OpenSplat. Otherwise, the original `gsplat` implementation is licensed under the Apache License v2.
