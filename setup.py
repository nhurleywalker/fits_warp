from distutils.core import setup

reqs = [
    "astropy",
    "numpy",
    "scipy",
    "matplotlib",
    "psutil",
    "tqdm",
]

setup(
    name="fits_warp",
    version="2.0",
    author="Natasha Hurley-Walker and Paul Hancock",
    description="Warp catalogues and images to remove the distorting affect of the ionosphere.",
    url="https://github.com/nhurleywalker/fits_warp",
    long_description=open("README.md").read(),
    packages=["fits_warp",],
    license="Academic Free License v3.0",
    requires=reqs,
    scripts=["fits_warp/fits_warp.py"],
)
