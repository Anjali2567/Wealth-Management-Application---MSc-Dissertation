    # # Train/Test Split (time series split: do not shuffle)
    # split_idx = int(len(combined_data) * 0.8)
    # X = combined_data.iloc[:-1]
    # y = combined_data.iloc[1:]
    # X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    # y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # # # Hyperparameter tuning for Random Forest
    # # param_grid = {
    # #     'n_estimators': [100, 200],
    # #     'max_depth': [3, 5, 10, None],
    # #     'min_samples_split': [2, 5, 10]
    # # }
    # # rf = RandomForestRegressor(random_state=42)


    # # # Define the hyperparameter grid for XGBoost
    # # param_grid = {
    # #     'n_estimators': [100, 200],
    # #     'max_depth': [3, 5, 10],
    # #     'learning_rate': [0.01, 0.1],
    # #     'subsample': [0.8, 1.0],
    # #     'colsample_bytree': [0.8, 1.0],
    # # }

    # # # Initialize XGBRegressor
    # # xgb = XGBRegressor(random_state=42, 
    # #                    objective='reg:squarederror', 
    # #                    n_jobs=-1, 
    # #                    learning_rate=0.01,
    # #                    n_estimators=1000,
    # #                    max_depth=3,
    # #                    subsample=0.8,
    # #                    colsample_bytree=0.8,
    # #                    tree_method='hist')

    # # Use time-series cross-validation
    # tscv = TimeSeriesSplit(n_splits=3, test_size=6)  # adjust test_size based on your data frequency

    # # Run GridSearchCV
    # grid_search = GridSearchCV(xgb, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    # grid_search.fit(X_train, y_train)

    # # Best model
    # model = grid_search.best_estimator_

    # # Evaluate
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print("Model Evaluation:")
    # print(f"Mean Squared Error: {mse:.6f}")
    # print(f"R^2 Score: {r2:.4f}")

    # # Predict Future Prices (scaled)
    # future_prices_scaled = model.predict([combined_data.iloc[-1]])
    # future_prices_scaled = future_prices_scaled.reshape(-1)
    # future_prices_scaled_df = pd.DataFrame([future_prices_scaled], columns=combined_data.columns)

    # # Inverse transform returns to get them back to original scale for MPT
    # predicted_returns_scaled = future_prices_scaled_df[asset_columns].iloc[0].values.reshape(1, -1)
    # predicted_returns = returns_scaler.inverse_transform(predicted_returns_scaled).flatten()
    # predicted_returns = pd.Series(predicted_returns, index=asset_columns)

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")