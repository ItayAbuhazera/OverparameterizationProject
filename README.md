# Overparameterization Project

This repository contains the implementation for the final project of the **Foundations of Overparameterized Machine Learning** course, Spring 2024. The project is based on the paper *"Understanding Deep Learning Requires Rethinking Generalization"* by Zhang et al. (ICLR, 2017). Our implementation focuses on investigating the effect of overparameterization and generalization in neural networks.

## Project Summary

The goal of this project is to extend the experimental results from the paper by exploring the impact of overparameterization on the generalization error in neural networks. We focus on the double descent phenomenon, which was not discussed in the original paper. Our experiments are conducted using a convolutional neural network (CNN) on a subset of the CIFAR-10 dataset.

For more details, please refer to the report included in this repository.

## Repository Contents

- **Train and Plot.ipynb**: A Jupyter notebook containing the code used for training and plotting the results of our experiments on the CIFAR-10 dataset.
- **report.pdf**: The project report detailing the summary, relation to the course material, experimental results, and the significance of our findings.
- **data/**: The directory containing preprocessed datasets used in our experiments (not included in this repository for space reasons).

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
   
   Then, open `Train and Plot.ipynb`.

5. **Dataset Preparation**:
   Ensure that you have the CIFAR-10 dataset (with a subset of the classes used in the project: 'car', 'frog', 'horse', 'ship'). If the dataset is not available, you may download and preprocess it directly in the notebook.

6. **Run the Notebook**:
   Follow the instructions in the notebook to execute the training and plotting of the results. The notebook is designed to demonstrate the double descent phenomenon with varying model sizes.

## Results

The main finding of the project is the observation of the double descent phenomenon when training overparameterized neural networks on the CIFAR-10 dataset. The results are visualized in the notebook, showing the relationship between the number of model parameters and the generalization performance.

## Challenges

During the project, several challenges were encountered, including selecting appropriate model sizes and training configurations that could run efficiently on local machines. We overcame these by reducing the dataset size and fine-tuning the models.

For a full discussion on the challenges and significance of the project, refer to the `report.pdf` file.

## Authors

- **Itay Abuhazira**
- **Raz Monsonego**

## License

This project is licensed under the MIT License.
