# Overparameterization Project

This repository contains the implementation for the final project of the **Foundations of Overparameterized Machine Learning** course, Spring 2024. The project is based on the paper *"Understanding Deep Learning Requires Rethinking Generalization"* by Zhang et al. (ICLR, 2017). Our implementation focuses on investigating the effect of overparameterization and generalization in neural networks.

## Project Summary

The goal of this project is to extend the experimental results from the paper by exploring the impact of overparameterization on the generalization error in neural networks. We focus on the double descent phenomenon, which was not discussed in the original paper. Our experiments are conducted using a convolutional neural network (CNN) on a subset of the CIFAR-10 dataset.

For more details, please refer to the report included in this repository.

## Repository Contents

- **train_plot.ipynb**: A Python notebook containing the code used for training, fine-tuning, and plotting the results of ALL of our experiments on the CIFAR-10 dataset. Notice that running this
notebook is resource demanding! We recommend running it on a machine with a GPU, or running single-model notebooks such as `cnn_loss_generalization_OML_19488.ipynb`.
- **cnn_loss_generalization_OML_19488.ipynb**: A Python notebook containing the code used for training a single CNN model on a subset of the CIFAR-10 dataset.
- **report.pdf**: The project report detailing the summary, relation to the course material, experimental results, and the significance of our findings.

## Requirements

The code in this repository is implemented using Python and PyTorch. Below are the main dependencies required to run the notebook:

- Python 3.7 or higher
- Jupyter Notebook
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

To install the required packages, you can run the following command:
```bash
pip install torch torchvision matplotlib numpy tqdm
```

## Running the Notebook

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ItayAbuhazera/OverparameterizationProject.git
   ```
   
   ```bash
   cd OverparameterizationProject
   ```

3. **Launch the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   
   Then, open `cnn_loss_generalization_OML_19488.ipynb` or  `train_plot.ipynb`.

5. **Dataset Preparation**:
   You don't need to manually download any dataset. The CIFAR-10 dataset will be automatically downloaded during the execution of the code. The notebook will further process and extract a subset containing the following classes: 'car', 'frog', 'horse', and 'ship'.

6. **Run the Notebook**:
   Simply run the cells of the notebook. The notebook is designed to demonstrate the double descent phenomenon with varying model sizes.

For a full discussion on the challenges and significance of the project, refer to the `report.pdf` file.

## Authors

- **Itay Abuhazera**
- **Raz Monsonego**

## License

This project is licensed under the MIT License.
