# Energy Efficiency in 5G Networks Using Gaussian Mixture Models

## Overview

This project applies Gaussian Mixture Models (GMM) to cluster synthetic 5G traffic patterns and simulate dynamic energy optimization.  

The model automatically determines the optimal number of traffic clusters using Bayesian Information Criterion (BIC) and compares performance against K-Means clustering using Silhouette Score.

---

## Problem Statement

5G networks experience fluctuating traffic demand. Static energy allocation leads to inefficiency and power wastage.  

This project models traffic distributions probabilistically and estimates adaptive energy allocation strategies based on traffic intensity clusters.

---

## Methodology

1. Synthetic traffic data generation (Low, Medium, High density)
2. Feature scaling using StandardScaler
3. Model selection using Bayesian Information Criterion (BIC)
4. Gaussian Mixture Model (GMM) clustering
5. K-Means clustering comparison
6. Evaluation using Silhouette Score
7. Energy estimation based on cluster intensity

---

## Model Selection Formula

Gaussian Mixture Models assume data is generated from a mixture of multiple Gaussian distributions:

p(x) = Σ π_k N(x | μ_k, Σ_k)

BIC is used to balance model fit and model complexity.

Silhouette Score is computed as:

s = (b - a) / max(a, b)

where:
- a = intra-cluster distance
- b = nearest-cluster distance

---

## Results

- Optimal number of clusters (BIC): 3  
- Silhouette Score ≈ 0.74  
- Both GMM and K-Means show strong cluster separation on well-structured synthetic data  
- Demonstrates effectiveness of probabilistic clustering for traffic modeling  

---

## Visualization

![Clustering Output](gmm_output.png)

---

## Repository Structure

- `main.py` → Standalone Python implementation  
- `5G_Energy_Efficiency_GMM.ipynb` → Research-style notebook  
- `gmm_output.png` → Visualization output  
- `IEEE Certificate.pdf` → Conference presentation proof  

---

## Technologies Used

- Python  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Jupyter Notebook  

---

## Conference Presentation

Presented at IEEE International Conference on Communication, Computing and Signal Processing (IICCCS), 2024.
