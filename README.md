# accimage benchmarks

## Set up

```bash
$ conda env create -n accimage-benchmark -f environment.yml
$ conda uninstall -y --force jpeg libtiff libjpeg-turbo
$ pip   uninstall -y         jpeg libtiff libjpeg-turbo
$ conda install -yc conda-forge libjpeg-turbo
$ CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
$ conda install -y jpeg libtiff
```