

Jagani, Sarthik, 22306164


Mangaroliya, Shubham, 2230621

# Hyundai Car Price Prediction System


https://mygit.th-deg.de/sm20211/Hyundai


https://mygit.th-deg.de/sm20211/Hyundai/-/wikis/home

## Project Description

This project predicts the price of Hyundai cars based on various features such as model, year, mileage, fuel type, and more. The system uses a machine learning model to analyze the data and make predictions. Additionally, it includes a chatbot for enhanced user interaction and a module to visualize original and augmented datasets.

## Installation

### Prerequisites

- Python 3.10+
- Flask
- Streamlit
- Scikit-learn
- Matplotlib
- Seaborn

### Steps

1. Clone the repository:
   ```
   git clone https://mygit.th-deg.de/sm20211/Hyundai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Linux/Mac:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. The code automatically reads the data from the provided file, so no manual placement of `hyundi.csv` in the root directory is needed.

### Running the Project

1. Train the Rasa model:
   ```
   rasa train
   ```

2. Start the Rasa server:
   ```
   rasa run
   ```

3. To run Action file:
   ```
   rasa run actions
   ```

4. Launch the Streamlit application:
   ```
   streamlit run main.py
   ```

5. Open the application in your browser at `localhost:8501`.

## Data

- **Source:** The primary dataset (`hyundi.csv`) contains Hyundai car data.
- **Preprocessing:** Features are standardized and encoded using Scikit-learn transformers.
- **Augmented Data:** Includes an extended dataset (`combined_hyundi_dataset_with_fake.csv`).

### Handling Outliers

Outliers are identified and removed using statistical analysis and domain-specific heuristics.

### Creating Fake Data

The augmented dataset includes synthetic data generated using data augmentation techniques to increase dataset diversity. Specifically, the Python module `random.choice(seq)` is used to generate synthetic data based on existing patterns in the dataset.

## Basic Usage

1. Navigate through the dashboard for data analysis, visualization, and predictions.
2. Use the chatbot to query specific details about Hyundai cars, such as price ranges or fuel efficiency.
3. Visualize and compare the distributions of real and augmented datasets.

## Work Done

- **Shubham Mangaroliya and Sarthik Jagani:** Jointly worked on preprocessing datasets, implementing visualization techniques, handling data augmentation, designing and implementing the Flask backend, and managing chatbot integration.

## Additional Details

This project employs a user-friendly interface and robust backend to provide accurate predictions and enhance user interaction. For more technical details, please refer to the project wiki.

## Screen recording video

1. **File Description**: This screen recording captures all tasks performed.  
2. **File Location**: The video is stored in the "Screen recording" file within the wiki repository for easy access.



