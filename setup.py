from setuptools import setup, find_packages

setup(
    name='dogretrieval',
    version='0.1',
    packages=find_packages(),  # 自动查找所有包（包括 Dogclip、DogUI 等）
    install_requires=[],       # 如果有依赖，比如 torch/clip，你也可以列在这里
)