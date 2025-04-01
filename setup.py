from setuptools import setup

setup(
    name='aime-api-worker-interface',
    version='0.9.9',
    author='AIME',
    author_email='carlo@aime.info',
    scripts=['aime_api_worker_interface/awi'],
    install_requires=[
        'pillow>=10.2.0',
        'requests>=2.31.0',
        'huggingface_hub>=0.28.0',
    ],
    packages=['aime_api_worker_interface'],
    zip_safe=False
)
