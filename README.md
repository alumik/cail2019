# CAIL2019-SCM: Similar Case Matching in Legal Domain

![version-1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
![python->=3.9,<3.13](https://img.shields.io/badge/python->=3.9,<3.13-blue?logo=python&logoColor=white)
![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)
![GitHub License](https://img.shields.io/github/license/alumik/cail2019)

Official GitHub repository for CAIL2019: https://github.com/china-ai-law-challenge/CAIL2019.

Dataset: https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip.

## Note

**Important:** Since `transformers` does not support Keras 3, one has to use `TF_USE_LEGACY_KERAS=1` to run the code.

We have changed the algorithm to siamese BERT network, instead of the naive BERT model in the original paper.

Our current accuracy on the test dataset is 0.6901.
