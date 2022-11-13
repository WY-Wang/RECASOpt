import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RECASOpt",
    version="0.0.2",
    author="Wenyu Wang",
    author_email="wenyu_wang@u.nus.edu",
    description="An surrogate-assisted optimizer for expensive many-objective optimization problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WY-Wang/RECASOpt",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy==1.21.1",
        "scipy==1.8.0",
    ],
)
