# Model Performance Evaluation Web App

## Description

This project is a simple Streamlit web application designed to evaluate the performance of a risk prediction model. It is intended for use with Mirai and Sybil. The application allows users to upload a results table, run evaluations, and generate performance reports in PDF format.

## Features

- Upload results table in CSV, TSV, XLS, or XLSX format.
- Evaluate model performance based on provided data.
- Generate and download a PDF report of the evaluation.
- Download overall metrics in CSV format.
- Display PDF report within the web app.

## Requirements

- Python 3.11 or above
- Streamlit
- Pandas
- Seaborn
- Matplotlib

## Installation

The following steps should be completed on a terminal.  

1. Clone the repository:
    ```sh
    git clone https://github.com/reginabarzilaygroup/GeneralEvaluation
    cd GeneralEvaluation
    ```

2. Create a virtual environment using the code below and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the package (which will also install required packages):
    ```sh
    pip install .
    ```
###  Install and run with pipx

You can also install the package using [pipx](https://pipx.pypa.io/stable/) to run the command-line tool without activating the virtual environment.
```shell
git clone https://github.com/reginabarzilaygroup/GeneralEvaluation
cd GeneralEvaluation
pipx install .
general-eval
```


## Run

1. Run the Streamlit app:
    ```sh
    streamlit run general_eval/app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Follow the instructions on the web app to upload your results table and run the evaluation.

# Usage

1. Prepare your results from Mirai or Sybil. See the [Input Data Format](#input-data-format) section for details.
2. Upload a results table in CSV, TSV, XLS, or XLSX format.
3. Adjust sensitivity and PPV targets as desired. (optional)
4. Click the "Run Evaluation" button.
5. Download the PDF report and the CSV file with overall metrics.

## Input Data Format

The input data should be a table with the following columns:
Upload a file containing the results of the model. The file should contain the following columns:   
 - `Days_to_Cancer`: Days between exam and cancer diagnosis. '-1' indicates no cancer diagnosis.  
 - `Days_Followup`: Days between exam and latest follow-up.   
 - `Year1`: Model prediction for year 1.  
 - `Year2`: Model prediction for year 2.  
 - `Year3`: Model prediction for year 3.  
 - `Year4`: Model prediction for year 4.  
 - `Year5`: Model prediction for year 5.  
 - `Year6`: Model prediction for year 6. (optional)

Remove PHI before uploading. If there are missing columns, you will see an error occur. Any additional columns will be ignored.

See the example data file in the `data` directory for reference.

## Example Data

An example data file is provided in the `data` folder. You can download it from the web app to see the expected format of the input data.

## Methods

The performance is evaluated using multiple metrics. Area-under-the-curve (AUC) and precision-recall curve (PRC) are calculated with [scikit-learn](https://scikit-learn.org/stable/) functions. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- Streamlit for providing an easy-to-use framework for building web apps.
- The developers of Mirai and Sybil for their contributions to risk prediction models.

## Troubleshooting

- **Global streamlit installation**
  - **Problem**: A global streamlit installation interferes with the local namespace, causing streamlit to fail at recognizing additional local dependencies. This might result in an error such as the following:
      ```
    ModuleNotFoundError: No module named 'matplotlib'
    Traceback:
    File "./GeneralEvaluation/general_eval/app.py", line 24, in <module>
    from general_eval.main import run_full_eval, DIAGNOSIS_DAYS_COL, FOLLOWUP_DAYS_COL
    File "./GeneralEvaluation/general_eval/main.py", line 14, in <module>
    from matplotlib import pyplot as plt
     ```
  - **Solution**: 
    -   Depending on your installation method, uninstall your global streamlit instance with ```pipx uninstall streamlit``` \
         or (in a new venv free terminal) ```pip uninstall streamlit```
    -   Re-install 'General Evaluation' ```pipx install . --force``` \
     or ```pip install .```
     

