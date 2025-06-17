# Mental Health ML

A machine learning project to predict mental health conditions in tech workplaces using a survey dataset. The project features a FastAPI backend for predictions, a Streamlit frontend for visualization, and is deployed on Azure App Service with a CI/CD pipeline via GitHub Actions.

[![GitHub Actions](https://github.com/sabeen864/mental-health-ml/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/sabeen864/mental-health-ml/actions)
[![Docker Hub](https://img.shields.io/docker/v/sabeen864/mental-health-ml-api?label=Docker%20Hub)](https://hub.docker.com/r/sabeen864/mental-health-ml-api)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project uses the "Mental Health in Tech Survey" dataset from Kaggle to predict mental health treatment needs in tech workplaces. It includes:
- A **FastAPI** backend serving a machine learning model (trained with RandomForest, LogisticRegression, XGBoost).
- A **Streamlit** frontend for interactive data visualization.
- **Azure Blob Storage** for storing datasets and models.
- **Docker** containers for packaging the API and frontend.
- **GitHub Actions** for automated testing, building, and deployment.
- **Azure App Service** for hosting the FastAPI app.

## Features
- Predict mental health treatment needs via API (`/predict`, `/health`).
- Interactive Streamlit dashboard for user input and results.
- Model training and tracking with MLflow.
- Secure storage of models and data in Azure Blob Storage.
- Automated CI/CD pipeline with GitHub Actions.
- Scalable deployment on Azure App Service.

## Tech Stack
- **Backend**: FastAPI, Python 3.10
- **Frontend**: Streamlit
- **ML**: pandas, scikit-learn, xgboost, MLflow
- **Storage**: Azure Blob Storage
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: Azure App Service
- **Dependencies**: See `requirements.txt`

## Setup Instructions
### Prerequisites
- Python 3.10
- Docker
- Git
- Azure account
- Docker Hub account
- Kaggle account (for dataset)

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/sabeen864/mental-health-ml.git
   cd mental-health-ml
