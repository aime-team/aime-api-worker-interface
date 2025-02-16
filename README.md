# AIME API Worker Interface

Interface library to connect deep learning PyTorch/TensorFlow model implementations to the [AIME API Server](https://github.com/aime-team/aime-api-server) as so-called compute workers to expose model functions as scalable and streamable HTTP/HTTPS API endpoints.

For more information about AIME API Server, please visit: [https://api.aime.info](https://api.aime.info)

For documentation on how to implement a Pytorch/Tensorflow worker for serving (your) models as scalable API endpoints, please read on here:

[AIME API Worker Interface Docs](https://api.aime.info/docs/api_worker_interface/index_api_worker_interface.html)

# Install AIME Worker Interface Pip

The AIME API worker interface is a Python Pip package. It currently can be installed through pip from GitHub with following command:

```
pip install git+https://github.com/aime-team/aime-api-worker-interface
```

# AWI Command

The aime-api-worker-interface pip provides also the AWI command, the central command to install, run and manage AIME API workers.

## AWI Download Model Weights

With the AWI command models can be downloaded from the Hugging Face model hub or cached versions from the AIME model storages.

Example to download Llama-3.3-70B-Instruct:

```
awi download-weights meta-llama/Llama-3.3-70B-Instruct
```
