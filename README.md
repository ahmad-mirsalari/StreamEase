
# StreamEase
A python library that automates the conversion of Temporal Convolutional Network (TCN) models to a streaming format without affecting model accuracy, significantly reducing deployment time and computational
resources.

## Reference
This library is based on the research outlined in the following paper:

<!-- - S. A. Mirsalari, L. Bijar, M. Fariselli, M. Croome, F. Paci, G. Tagliavini, L. Benini , "StreamEase: Enabling Real-Time Inference of Temporal Convolution Networks on Low-Power MCUs with Stream-Oriented Automatic Transformation", 2024 31th IEEE International Conference on Electronics, Circuits and Systems (ICECS), [Link to the paper](https://ieeexplore.ieee.org/abstract/document/10136916) -->

<!-- If you find this library useful in your research, please consider citing the paper: -->



## Setup
#### Clone the Repository:
~~~~~shell
git clone git@github.com:ahmad-mirsalari/TCN_Pre_release.git
cd TCN_Pre_release
~~~~~

#### Install Dependencies:
<!-- To ensure compatibility when checking network functionality, particularly with onnxruntime, I recommend using Python 3.10. This version has been found to work well with onnxruntime, avoiding some version conflicts that might occur with newer Python versions. -->
You can install the required dependencies by running:
~~~~~shell
pip install -r requirements.txt
~~~~~
#### Set Up the tool:
Run the setup_env.sh script to add the project root to your PYTHONPATH. This will ensure that Python can locate the streamease package.
~~~~~shell
chmod +x setup_env.sh
./setup_env.sh
~~~~~
You will be prompted to add the PYTHONPATH to your .bashrc to make it permanent for future sessions.

## Repository Organization
- [streamease](./streamease/): Main package containing modules for streaming inference, buffering, and quantization.
- [examples](./examples/): Contains example scripts demonstrating how to use the toolkit.
- [utils](./utils/): Contains helper functions to facilitate running the streaming network, with support for ONNX Runtime and GreenWaves Technologies' nntool for optimized model execution.
- [setup_env.sh](setup_env.sh): Script for setting up dependencies and environment variables.
- [requirements.txt](requirements.txt): List of required dependencies

## Roadmap

- Extend the library by adding additional networks in different applications


## License 
 StreamEase is released under Apache 2.0, see the [LICENSE](./LICENSE.md) file in the root of this repository for details.

## Acknowledgements
This work was supported by the [APROPOS](https://projects.tuni.fi/apropos/) project (g.a. no. 956090), founded by the European Union’s Horizon 2020 research and innovation program. 


## Contributors
- [Seyed Ahmad Mirsalari](https://github.com/ahmad-mirsalari), University of Bologna,[E-mail](mailto:seyedahmad.mirsalar2@unibo.it)


## 🚀 Contact Me
- [Email](mailto:seyedahmad.mirsalar2@unibo.it)
- [LinkedIn](https://www.linkedin.com/in/ahmad-mirsalari/)
- [Twitter](https://twitter.com/ahmad_mirsalari)
- [APROPOS](https://projects.tuni.fi/apropos/news/pr_esr_3/)
