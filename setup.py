from setuptools import setup, find_packages

VERSION = "1.0.0"

with open("README.md") as readme_file:
    readme = readme_file.read()

install_requires = [
    "torch>=1.9.0",
    "pytorch-lightning~=1.4.2",
    "torchmetrics~=0.5.0",
    "tqdm~=4.62.1",
    "wandb~=0.12.0",
    "omegaconf~=2.1.1",
    "commode-utils>=0.3.8",
]

setup_args = dict(
    name="code2seq",
    version=VERSION,
    description="Set of pytorch modules and utils to train code2seq model",
    long_description_content_type="text/markdown",
    long_description=readme,
    install_requires=install_requires,
    license="MIT",
    packages=find_packages(),
    author="Egor Spirin",
    author_email="spirin.egor@gmail.com",
    keywords=["code2seq", "pytorch", "pytorch-lightning", "ml4code", "ml4se"],
    url="https://github.com/JetBrains-Research/code2seq",
    download_url="https://pypi.org/project/code2seq/",
)

if __name__ == "__main__":
    setup(**setup_args)
