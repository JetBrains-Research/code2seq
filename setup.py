from setuptools import setup, find_packages

VERSION = "0.0.4"

with open("README.md") as readme_file:
    readme = readme_file.read()

install_requires = [
    "torch~=1.7.1",
    "tqdm~=4.58.0",
    "numpy~=1.20.1",
    "pytorch-lightning==1.1.7",
    "wandb~=0.10.20",
    "hydra-core~=1.0.6",
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
