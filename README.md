# IPL Player Performance Predictor

## Overview
This project predicts the future performance of IPL (Indian Premier League) players—both batsmen and bowlers—using historical match data and machine learning models. It provides predictions for runs, averages, wickets, and other key statistics for upcoming matches.

## Project Structure
- **IPL_Player_Performance_Predictor.py**: Main Python script containing the data processing, feature engineering, model training, and prediction logic for both batsmen and bowlers.
- **IPL Complete Dataset (2008-2024).csv**: Large dataset containing ball-by-ball IPL match data from 2008 to 2024.
- **Batsmen_Predictions.csv**: Output file with predicted statistics for all batsmen in the dataset.
- **Bowlers_Predictions.csv**: Output file with predicted statistics for all bowlers in the dataset.
- **Research Paper.docx**: Research paper related to the project.
- **myenv/**: Python virtual environment directory (auto-generated, not project-specific).

## Main Features
- Loads and preprocesses IPL match data.
- Generates advanced statistics for batsmen and bowlers (e.g., average, strike rate, economy, consistency, recent form).
- Trains Random Forest models to predict future performance for each player.
- Exports predictions for all players to CSV files.

## How to Use
1. **Install dependencies** (recommended: use a virtual environment):
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. **Place the IPL dataset** as `IPL Complete Dataset (2008-2024).csv` in the project directory.
3. **Run the main script**:
   ```bash
   python IPL_Player_Performance_Predictor.py
   ```
   This will generate `Batsmen_Predictions.csv` and `Bowlers_Predictions.csv` in the same directory.

## Output Files
- **Batsmen_Predictions.csv**: Contains columns such as batsman name, matches analyzed, total runs, average, strike rate, boundary percentage, consistency score, recent form, predicted runs for next 5 matches, predicted average, and confidence intervals.
- **Bowlers_Predictions.csv**: Contains columns such as bowler name, matches analyzed, total wickets, economy, average, strike rate, consistency score, recent form, predicted wickets for next 5 matches, predicted economy, and confidence intervals.

## Notes
- The main dataset file is very large and may require significant memory to process.
- The script is designed to be run as a standalone program and will automatically generate predictions for all players.
- For more details on the methodology, refer to the included research paper (if available).

## License
This project is for educational and research purposes only. Please cite appropriately if using in academic work. 