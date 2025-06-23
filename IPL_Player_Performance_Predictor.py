_n = "batting"
_m = "recent_form"
_l = "confidence_interval"
_k = "future_matches"
_j = "consistency_score"
_i = "prediction"
_h = "matches_analyzed"
_g = "ignore"
_f = "extras"
_e = "nunique"
_d = "Data not loaded. Call load_and_preprocess() first."
_c = "extra_runs"
_b = "overs"
_a = "dismissals"
_Z = "sixes"
_Y = "fours"
_X = "current_statistics"
_W = "error"
_V = "runs_conceded"
_U = "left"
_T = "boundary_percentage"
_S = "ball"
_R = "balls_bowled"
_Q = False
_P = "balls_faced"
_O = "sum"
_N = "consistency"
_M = "is_wicket"
_L = "wickets"
_K = "batsman_runs"
_J = "economy"
_I = "average"
_H = "strike_rate"
_G = "matches_played"
_F = "match_id"
_E = "total_runs"
_D = "form"
_C = "batter"
_B = "bowler"
_A = None
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings, os, logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IPLPerformanceAnalyzer:
    def __init__(self, file_path=_A, data=_A):
        self.file_path = file_path
        self.df = data
        self.batsman_stats = _A
        self.bowler_stats = _A
        self.batsman_model = _A
        self.bowler_model = _A
        self.batsman_scaler = StandardScaler()
        self.bowler_scaler = StandardScaler()
        if file_path is _A and data is _A:
            raise ValueError("Either file_path or data must be provided")
        if data is not _A and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

    def load_and_preprocess(self):
        C = "not_dismissed"
        B = "unknown"
        A = "int8"
        if self.df is not _A:
            logger.info("Using pre-loaded DataFrame")
            return self
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            logger.info(f"Loading data from {self.file_path}")
            dtypes = {
                _F: "int32",
                "inning": A,
                "over": A,
                _S: A,
                _K: A,
                _c: A,
                _E: A,
                _M: A,
            }
            self.df = pd.read_csv(self.file_path, dtype=dtypes, low_memory=True)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise e
            else:
                raise ValueError(f"Error loading data: {str(e)}")
        required_columns = [_F, _K, _C, _B, _M]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {", ".join(missing_columns)}")
        categorical_defaults = {
            "player_dismissed": C,
            "dismissal_kind": C,
            "fielder": "no_fielder",
            "extras_type": "no_extras",
            "batting_team": B,
            "bowling_team": B,
            _C: B,
            _B: B,
        }
        for col, default_val in categorical_defaults.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(default_val).astype("category")
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        self.df[numerical_cols] = self.df[numerical_cols].fillna(0)
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        return self

    def _generate_batsman_stats(self):
        if self.df is _A or self.df.empty:
            raise ValueError(_d)
        logger.info("Generating batsman statistics")
        agg_dict = {
            _K: [_O, lambda x: (x == 4).sum(), lambda x: (x == 6).sum()],
            _S: "count",
            _F: _e,
            _M: _O,
        }
        batsman_stats = self.df.groupby(_C).agg(agg_dict)
        batsman_stats.columns = [_E, _Y, _Z, _P, _G, _a]
        batsman_stats = batsman_stats.reset_index()
        batsman_stats = batsman_stats[batsman_stats[_P] >= 10]
        batsman_stats[_H] = batsman_stats[_E] / batsman_stats[_P] * 100
        batsman_stats[_I] = batsman_stats[_E] / np.maximum(batsman_stats[_a], 1)
        batsman_stats[_T] = (
            (batsman_stats[_Y] + batsman_stats[_Z])
            / np.maximum(batsman_stats[_P], 1)
            * 100
        )
        runs_per_match = self.df.groupby([_C, _F])[_K].sum().reset_index()
        consistency_std = runs_per_match.groupby(_C)[_K].std().fillna(0)
        consistency_mean = runs_per_match.groupby(_C)[_K].mean()
        consistency_cv = consistency_std / np.maximum(consistency_mean, 1)
        consistency = (1 - consistency_cv).clip(0, 1).reset_index()
        consistency.columns = [_C, _N]
        recent_matches = self.df.sort_values(_F, ascending=_Q)
        recent_perfomance = recent_matches.groupby([_C, _F])[_K].sum().reset_index()
        form_data = []
        for player in batsman_stats[_C].unique():
            player_recent = recent_perfomance[recent_perfomance[_C] == player].head(5)
            if not player_recent.empty:
                weighted_form = np.average(
                    player_recent[_K], weights=np.linspace(1, 0.6, len(player_recent))
                )
                form_data.append({_C: player, _D: weighted_form})
        form_df = pd.DataFrame(form_data)
        batsman_stats = pd.merge(batsman_stats, consistency, on=_C, how=_U)
        batsman_stats = pd.merge(batsman_stats, form_df, on=_C, how=_U)
        batsman_stats[_D] = batsman_stats[_D].fillna(
            batsman_stats[_E] / batsman_stats[_G]
        )
        self.batsman_stats = batsman_stats
        logger.info(f"Generated statistics for {len(batsman_stats)} batsmen")
        return self.batsman_stats

    def _generate_bowler_stats(self):
        if self.df is _A or self.df.empty:
            raise ValueError(_d)
        logger.info("Generating bowler statistics")
        agg_dict = {_M: _O, _S: "count", _E: _O, _F: _e, _c: _O}
        bowler_stats = self.df.groupby(_B).agg(agg_dict)
        bowler_stats.columns = [_L, _R, _V, _G, _f]
        bowler_stats = bowler_stats.reset_index()
        bowler_stats = bowler_stats[bowler_stats[_R] >= 30]
        bowler_stats[_b] = bowler_stats[_R] / 6
        bowler_stats[_J] = bowler_stats[_V] / np.maximum(bowler_stats[_b], 1)
        bowler_stats[_I] = bowler_stats[_V] / np.maximum(bowler_stats[_L], 1)
        bowler_stats[_H] = bowler_stats[_R] / np.maximum(bowler_stats[_L], 1)
        bowler_stats["wickets_per_match"] = bowler_stats[_L] / bowler_stats[_G]
        economy_per_match = (
            self.df.groupby([_B, _F])
            .apply(lambda x: x[_E].sum() / np.maximum(x[_S].count() / 6, 1))
            .reset_index(name=_J)
        )
        eco_std = economy_per_match.groupby(_B)[_J].std().fillna(0)
        eco_mean = economy_per_match.groupby(_B)[_J].mean()
        eco_cv = eco_std / np.maximum(eco_mean, 1)
        consistency = (1 - eco_cv).clip(0, 1).reset_index()
        consistency.columns = [_B, _N]
        recent_matches = self.df.sort_values(_F, ascending=_Q)
        wickets_per_match = recent_matches.groupby([_B, _F])[_M].sum().reset_index()
        form_data = []
        for player in bowler_stats[_B].unique():
            player_recent = wickets_per_match[wickets_per_match[_B] == player].head(5)
            if not player_recent.empty:
                weighted_form = np.average(
                    player_recent[_M], weights=np.linspace(1, 0.6, len(player_recent))
                )
                form_data.append({_B: player, _D: weighted_form})
        form_df = pd.DataFrame(form_data)
        bowler_stats = pd.merge(bowler_stats, consistency, on=_B, how=_U)
        bowler_stats = pd.merge(bowler_stats, form_df, on=_B, how=_U)
        bowler_stats[_D] = bowler_stats[_D].fillna(bowler_stats[_L] / bowler_stats[_G])
        self.bowler_stats = bowler_stats
        logger.info(f"Generated statistics for {len(bowler_stats)} bowlers")
        return self.bowler_stats

    def _train_batsman_model(self):
        if self.batsman_stats is _A:
            self._generate_batsman_stats()
        if len(self.batsman_stats) < 10:
            raise ValueError("Insufficient data to train model (fewer than 10 batsmen)")
        logger.info("Training batsman performance model")
        feature_cols = [_Y, _Z, _P, _G, _a, _H, _I, _T, _N, _D]
        available_features = [
            col for col in feature_cols if col in self.batsman_stats.columns
        ]
        features = self.batsman_stats[available_features].copy()
        target = self.batsman_stats[_E].values
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        X_train_scaled = self.batsman_scaler.fit_transform(X_train)
        X_val_scaled = self.batsman_scaler.transform(X_val)
        self.batsman_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter(_g)
            self.batsman_model.fit(X_train_scaled, y_train)
        train_score = self.batsman_model.score(X_train_scaled, y_train)
        val_score = self.batsman_model.score(X_val_scaled, y_val)
        logger.info(
            f"Batsman model R² scores - Train: {train_score:.4f}, Validation: {val_score:.4f}"
        )
        self.batsman_feature_names = available_features
        return self

    def _train_bowler_model(self):
        if self.bowler_stats is _A:
            self._generate_bowler_stats()
        if len(self.bowler_stats) < 10:
            raise ValueError("Insufficient data to train model (fewer than 10 bowlers)")
        logger.info("Training bowler performance model")
        feature_cols = [_R, _V, _G, _f, _b, _J, _I, _H, _N, _D]
        available_features = [
            col for col in feature_cols if col in self.bowler_stats.columns
        ]
        features = self.bowler_stats[available_features].copy()
        target = self.bowler_stats[_L].values
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        X_train_scaled = self.bowler_scaler.fit_transform(X_train)
        X_val_scaled = self.bowler_scaler.transform(X_val)
        self.bowler_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter(_g)
            self.bowler_model.fit(X_train_scaled, y_train)
        train_score = self.bowler_model.score(X_train_scaled, y_train)
        val_score = self.bowler_model.score(X_val_scaled, y_val)
        logger.info(
            f"Bowler model R² scores - Train: {train_score:.4f}, Validation: {val_score:.4f}"
        )
        self.bowler_feature_names = available_features
        return self

    def predict_batsman_performance(self, batsman_name, future_matches=5):
        if self.batsman_stats is _A:
            self._generate_batsman_stats()
        if self.batsman_model is _A:
            self._train_batsman_model()
        logger.info(f"Predicting performance for batsman: {batsman_name}")
        if batsman_name not in self.batsman_stats[_C].values:
            return {_W: f"Batsman {batsman_name} not found in the dataset"}
        batsman_data = self.batsman_stats[self.batsman_stats[_C] == batsman_name]
        matches = batsman_data[_G].values[0]
        features = batsman_data[self.batsman_feature_names].copy()
        features_scaled = self.batsman_scaler.transform(features)
        predicted_runs = (
            self.batsman_model.predict(features_scaled)[0] / matches * future_matches
        )
        predictions = []
        for estimator in self.batsman_model.estimators_:
            pred = estimator.predict(features_scaled)[0] / matches * future_matches
            predictions.append(pred)
        lower_ci = np.percentile(predictions, 25)
        upper_ci = np.percentile(predictions, 75)
        result = {
            "batsman": batsman_name,
            _h: int(matches),
            _X: {
                _E: int(batsman_data[_E].values[0]),
                _I: round(batsman_data[_I].values[0], 2),
                _H: round(batsman_data[_H].values[0], 2),
                _T: round(batsman_data[_T].values[0], 2),
                _j: round(batsman_data[_N].values[0], 2),
            },
            _i: {
                _k: future_matches,
                "predicted_runs": round(predicted_runs, 1),
                "predicted_average": round(predicted_runs / future_matches, 2),
                _l: {"lower": round(lower_ci, 1), "upper": round(upper_ci, 1)},
            },
        }
        if _D in batsman_data.columns:
            result[_X][_m] = round(batsman_data[_D].values[0], 2)
        return result

    def predict_bowler_performance(self, bowler_name, future_matches=5):
        if self.bowler_stats is _A:
            self._generate_bowler_stats()
        if self.bowler_model is _A:
            self._train_bowler_model()
        logger.info(f"Predicting performance for bowler: {bowler_name}")
        if bowler_name not in self.bowler_stats[_B].values:
            return {_W: f"Bowler {bowler_name} not found in the dataset"}
        bowler_data = self.bowler_stats[self.bowler_stats[_B] == bowler_name]
        matches = bowler_data[_G].values[0]
        features = bowler_data[self.bowler_feature_names].copy()
        features_scaled = self.bowler_scaler.transform(features)
        predicted_wickets = (
            self.bowler_model.predict(features_scaled)[0] / matches * future_matches
        )
        predictions = []
        for estimator in self.bowler_model.estimators_:
            pred = estimator.predict(features_scaled)[0] / matches * future_matches
            predictions.append(pred)
        lower_ci = np.percentile(predictions, 25)
        upper_ci = np.percentile(predictions, 75)
        result = {
            _B: bowler_name,
            _h: int(matches),
            _X: {
                "total_wickets": int(bowler_data[_L].values[0]),
                _J: round(bowler_data[_J].values[0], 2),
                _I: round(bowler_data[_I].values[0], 2),
                _H: round(bowler_data[_H].values[0], 2),
                _j: round(bowler_data[_N].values[0], 2),
            },
            _i: {
                _k: future_matches,
                "predicted_wickets": round(predicted_wickets, 1),
                "predicted_economy": round(bowler_data[_J].values[0], 2),
                _l: {"lower": round(lower_ci, 1), "upper": round(upper_ci, 1)},
            },
        }
        if _D in bowler_data.columns:
            result[_X][_m] = round(bowler_data[_D].values[0], 2)
        return result

    def get_top_performers(self, category=_n, metric=_A, top_n=10):
        if category.lower() == _n:
            if self.batsman_stats is _A:
                self._generate_batsman_stats()
            stats = self.batsman_stats.copy()
            if metric is _A:
                metric = _E
            if metric not in stats.columns:
                raise ValueError(f"Metric '{metric}' not found in batsman statistics")
            top_players = stats.sort_values(metric, ascending=_Q).head(top_n)
            return top_players.loc[
                :,
                [_C, metric]
                + [col for col in stats.columns if col != _C and col != metric],
            ]
        elif category.lower() == "bowling":
            if self.bowler_stats is _A:
                self._generate_bowler_stats()
            stats = self.bowler_stats.copy()
            if metric is _A:
                metric = _L
            ascending = metric in [_J, _I, _H]
            if metric not in stats.columns:
                raise ValueError(f"Metric '{metric}' not found in bowler statistics")
            top_players = stats.sort_values(metric, ascending=ascending).head(top_n)
            return top_players.loc[
                :,
                [_B, metric]
                + [col for col in stats.columns if col != _B and col != metric],
            ]
        else:
            raise ValueError("Category must be either 'batting' or 'bowling'")


def export_all_players(file_path):
    try:
        logger.info("Initializing IPL Performance Analyzer")
        analyzer = IPLPerformanceAnalyzer(file_path)
        analyzer.load_and_preprocess()
        logger.info("Generating batsmen predictions")
        batsmen_stats = analyzer._generate_batsman_stats()
        analyzer._train_batsman_model()
        all_batsmen = []
        for batsman in batsmen_stats[_C].unique():
            pred = analyzer.predict_batsman_performance(batsman, 5)
            if _W not in pred:
                all_batsmen.append(pred)
        batsmen_df = pd.json_normalize(all_batsmen)
        batsmen_output = "Batsmen_Predictions.csv"
        batsmen_df.to_csv(batsmen_output, index=_Q)
        logger.info("Generating bowlers predictions")
        bowlers_stats = analyzer._generate_bowler_stats()
        analyzer._train_bowler_model()
        all_bowlers = []
        for bowler in bowlers_stats[_B].unique():
            pred = analyzer.predict_bowler_performance(bowler, 5)
            if _W not in pred:
                all_bowlers.append(pred)
        bowlers_df = pd.json_normalize(all_bowlers)
        bowlers_output = "Bowlers_Predictions.csv"
        bowlers_df.to_csv(bowlers_output, index=_Q)
        logger.info(
            f"Successfully exported predictions for {len(all_batsmen)} batsmen and {len(all_bowlers)} bowlers"
        )
        return batsmen_output, bowlers_output
    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        file_path = "IPL Complete Dataset (2008-2024).csv"
        batsmen_file, bowlers_file = export_all_players(file_path)
        print(f"Exported batsmen predictions to: {batsmen_file}")
        print(f"Exported bowlers predictions to: {bowlers_file}")
    except Exception as e:
        print(f"Error: {str(e)}")