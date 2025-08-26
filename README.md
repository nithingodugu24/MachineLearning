# Machine Learning Study Workspace

This repository is a curated workspace of Jupyter notebooks and small demos covering foundational machine learning topics: preprocessing, linear/logistic regression, regularization, trees, ensembles, clustering, gradient descent variants, and a few end-to-end mini projects.

## How to use this repo
- Browse topic folders and open notebooks (`.ipynb`) to run cells in order.
- Many folders include datasets (`.csv`, `.pkl`) used by notebooks; paths are relative to each notebook.
- Feel free to copy notebooks to experiment; keep dataset paths consistent.

## Environment setup
1. Install Python 3.9+ and pip.
2. Create a virtual environment and install basics:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -U pip jupyterlab numpy pandas scikit-learn matplotlib seaborn plotly
   ```
3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```

## Topics overview
- PreProcessing: EDA, scaling, PCA, and profiling.
- Linear Regression: simple and multiple regression, metrics.
- Logistic Regression: sigmoid, softmax, perceptron trick, metrics.
- Regularization: Ridge and ElasticNet.
- Decision Trees and RandomForest: classification/regression, feature importance, tuning.
- Ensemble: bagging, boosting (AdaBoost, Gradient Boosting), voting.
- Clustering: K-Means examples.
- Gradient Descent: batch, stochastic, and mini-batch implementations.
- Projects: recommender app and other small applied case studies.

## Repository structure
The top-level folders each correspond to a concept area. Each area includes one or more notebooks and supporting data files. Subfolders may include focused demos (e.g., `K-Means` under `Clustering`, `EDA` under `PreProcessing`).

## Conventions
- Notebooks aim to be runnable as-is with minimal configuration.
- Where helpful, introductory markdown cells will describe the objective, dataset, and steps.
- File paths are relative. If loading issues occur, ensure you started Jupyter in the `ml` directory.

## Notes
- Large binary artifacts (e.g., `.pkl`) are included for convenience in demos that load precomputed models or processed data.
- Some projects include simple web apps (Flask) for demonstration.

Enjoy learning and experimenting!
