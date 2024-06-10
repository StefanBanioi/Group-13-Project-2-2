{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import xgboost as xgb\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to the CSV file\n",
    "csv_file_path = 'healthinsurance2.csv'  # Replace with the actual path to your CSV file\n",
    "# Read the CSV file\n",
    "mydata = pd.read_csv(csv_file_path)\n",
    "mydata = mydata.dropna()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = mydata.drop(columns=['claim','job_title','city'])\n",
    "y = mydata['claim']\n",
    "\n",
    "\n",
    "# X = data.drop('PremiumPrice', axis=1)\n",
    "# y = data['PremiumPrice']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_scaled = scaler.transform(X)\n",
    "X_train_scaled = scaler.transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "             gamma=0.1, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.1, max_bin=None,\n",
       "             max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=9, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=200, n_jobs=None,\n",
       "             num_parallel_tree=None, random_state=69, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">XGBRegressor</label><div class=\"sk-toggleable__content\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "             gamma=0.1, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.1, max_bin=None,\n",
       "             max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=9, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=200, n_jobs=None,\n",
       "             num_parallel_tree=None, random_state=69, ...)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "             gamma=0.1, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.1, max_bin=None,\n",
       "             max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=9, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=200, n_jobs=None,\n",
       "             num_parallel_tree=None, random_state=69, ...)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create an instance of XGBRegressor with the given parameters\n",
    "model = xgb.XGBRegressor(\n",
    "        objective='reg:squarederror',\n",
    "        tree_method='hist',  # Comment out if no GPU is available\n",
    "        max_depth=9,\n",
    "        gamma=0.1,\n",
    "        learning_rate=0.1,\n",
    "        n_estimators=200,\n",
    "        subsample=0.7,\n",
    "        random_state = 69)\n",
    "# possibleParameters= {\n",
    "#   'max_depth' : range (2, 51, 1),\n",
    "#   'n_estimators'  : range (50,310,5),\n",
    "#   'learning_rate' : [0.01, 0.02,0.03,0.04,0.05,0.1],\n",
    "#   'subsample':  [0.65,0.7,0.75,0.8,0.85,0.95,1],\n",
    "#   'gamma': [0,0.01,0.05,0.1]\n",
    "# }\n",
    "\n",
    "\n",
    "# rand_search = RandomizedSearchCV(\n",
    "#     xgb.XGBRegressor(random_state=69),\n",
    "#     param_distributions=possibleParameters,\n",
    "#     n_iter=1000,  # Set the number of iterations to 1\n",
    "#     cv=5, n_jobs=-1\n",
    "# )\n",
    "# rand_search.fit(X_train_scaled, y_train)\n",
    "\n",
    "# best_params = rand_search.best_params_\n",
    "\n",
    "# print(\"Best Parameters:\", best_params)\n",
    "\n",
    "\n",
    "# #best parameters\n",
    "\n",
    "\n",
    "# #predictions using the best model\n",
    "# model = rand_search.best_estimator_\n",
    "# Fit the model to the training data\n",
    "model.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[11264.904   7200.988  28355.445  ...  1154.8145  7311.284  18240.498 ]\n"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(X_test_scaled)\n",
    "print(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "358.576513385661\n",
      "2062.6248617684423\n",
      "0.02842792074167711\n"
     ]
    }
   ],
   "source": [
    "import math\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.metrics import mean_absolute_percentage_error\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "mape = mean_absolute_percentage_error(y_test, y_pred)\n",
    "print(mae)\n",
    "print(math.sqrt(mse))\n",
    "print(mape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import BaggingRegressor\n",
    "bagging_model = BaggingRegressor(\n",
    "    base_estimator=model,\n",
    "    n_estimators=10,       # Number of bootstrap samples\n",
    "    bootstrap=True,        # Use bootstrapped samples\n",
    "    n_jobs=-1,\n",
    "    random_state = 69# Use all available cores\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>BaggingRegressor(base_estimator=XGBRegressor(base_score=None, booster=None,\n",
       "                                             callbacks=None,\n",
       "                                             colsample_bylevel=None,\n",
       "                                             colsample_bynode=None,\n",
       "                                             colsample_bytree=None, device=None,\n",
       "                                             early_stopping_rounds=None,\n",
       "                                             enable_categorical=False,\n",
       "                                             eval_metric=None,\n",
       "                                             feature_types=None, gamma=0.1,\n",
       "                                             grow_policy=None,\n",
       "                                             importance_type=None,\n",
       "                                             interaction_constraints=None,\n",
       "                                             learning_rate=0.1, max_bin=None,\n",
       "                                             max_cat_threshold=None,\n",
       "                                             max_cat_to_onehot=None,\n",
       "                                             max_delta_step=None, max_depth=9,\n",
       "                                             max_leaves=None,\n",
       "                                             min_child_weight=None, missing=nan,\n",
       "                                             monotone_constraints=None,\n",
       "                                             multi_strategy=None,\n",
       "                                             n_estimators=200, n_jobs=None,\n",
       "                                             num_parallel_tree=None,\n",
       "                                             random_state=69, ...),\n",
       "                 n_jobs=-1, random_state=69)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" ><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">BaggingRegressor</label><div class=\"sk-toggleable__content\"><pre>BaggingRegressor(base_estimator=XGBRegressor(base_score=None, booster=None,\n",
       "                                             callbacks=None,\n",
       "                                             colsample_bylevel=None,\n",
       "                                             colsample_bynode=None,\n",
       "                                             colsample_bytree=None, device=None,\n",
       "                                             early_stopping_rounds=None,\n",
       "                                             enable_categorical=False,\n",
       "                                             eval_metric=None,\n",
       "                                             feature_types=None, gamma=0.1,\n",
       "                                             grow_policy=None,\n",
       "                                             importance_type=None,\n",
       "                                             interaction_constraints=None,\n",
       "                                             learning_rate=0.1, max_bin=None,\n",
       "                                             max_cat_threshold=None,\n",
       "                                             max_cat_to_onehot=None,\n",
       "                                             max_delta_step=None, max_depth=9,\n",
       "                                             max_leaves=None,\n",
       "                                             min_child_weight=None, missing=nan,\n",
       "                                             monotone_constraints=None,\n",
       "                                             multi_strategy=None,\n",
       "                                             n_estimators=200, n_jobs=None,\n",
       "                                             num_parallel_tree=None,\n",
       "                                             random_state=69, ...),\n",
       "                 n_jobs=-1, random_state=69)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">base_estimator: XGBRegressor</label><div class=\"sk-toggleable__content\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "             gamma=0.1, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.1, max_bin=None,\n",
       "             max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=9, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=200, n_jobs=None,\n",
       "             num_parallel_tree=None, random_state=69, ...)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">XGBRegressor</label><div class=\"sk-toggleable__content\"><pre>XGBRegressor(base_score=None, booster=None, callbacks=None,\n",
       "             colsample_bylevel=None, colsample_bynode=None,\n",
       "             colsample_bytree=None, device=None, early_stopping_rounds=None,\n",
       "             enable_categorical=False, eval_metric=None, feature_types=None,\n",
       "             gamma=0.1, grow_policy=None, importance_type=None,\n",
       "             interaction_constraints=None, learning_rate=0.1, max_bin=None,\n",
       "             max_cat_threshold=None, max_cat_to_onehot=None,\n",
       "             max_delta_step=None, max_depth=9, max_leaves=None,\n",
       "             min_child_weight=None, missing=nan, monotone_constraints=None,\n",
       "             multi_strategy=None, n_estimators=200, n_jobs=None,\n",
       "             num_parallel_tree=None, random_state=69, ...)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "BaggingRegressor(base_estimator=XGBRegressor(base_score=None, booster=None,\n",
       "                                             callbacks=None,\n",
       "                                             colsample_bylevel=None,\n",
       "                                             colsample_bynode=None,\n",
       "                                             colsample_bytree=None, device=None,\n",
       "                                             early_stopping_rounds=None,\n",
       "                                             enable_categorical=False,\n",
       "                                             eval_metric=None,\n",
       "                                             feature_types=None, gamma=0.1,\n",
       "                                             grow_policy=None,\n",
       "                                             importance_type=None,\n",
       "                                             interaction_constraints=None,\n",
       "                                             learning_rate=0.1, max_bin=None,\n",
       "                                             max_cat_threshold=None,\n",
       "                                             max_cat_to_onehot=None,\n",
       "                                             max_delta_step=None, max_depth=9,\n",
       "                                             max_leaves=None,\n",
       "                                             min_child_weight=None, missing=nan,\n",
       "                                             monotone_constraints=None,\n",
       "                                             multi_strategy=None,\n",
       "                                             n_estimators=200, n_jobs=None,\n",
       "                                             num_parallel_tree=None,\n",
       "                                             random_state=69, ...),\n",
       "                 n_jobs=-1, random_state=69)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bagging_model.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred2 = bagging_model.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n",
      "385.5439254707077\n",
      "2034.697367875924\n",
      "0.03430385968936558\n"
     ]
    }
   ],
   "source": [
    "mae = mean_absolute_error(y_test, y_pred2)\n",
    "mse = mean_squared_error(y_test, y_pred2)\n",
    "mape = mean_absolute_percentage_error(y_test, y_pred2)\n",
    "print(mae)\n",
    "print(math.sqrt(mse))\n",
    "print(mape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n",
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Cross-validated MAE: 374.82765058768564\n",
      "Mean Cross-validated RMSE: 1989.4273493719988\n",
      "Mean Cross-validated MAPE: 0.030367679635086596\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import cross_val_score, KFold\n",
    "kf = KFold(n_splits=10, shuffle=True, random_state=69)\n",
    "cv_mae = []\n",
    "cv_mse = []\n",
    "cv_mape = []\n",
    "for train_index, test_index in kf.split(X_scaled):\n",
    "    X_train, X_test = X_scaled[train_index], X_scaled[test_index]\n",
    "    y_train, y_test = y.iloc[train_index], y.iloc[test_index]\n",
    "\n",
    "    # Fit the BaggingRegressor model\n",
    "    bagging_model.fit(X_train, y_train)\n",
    "\n",
    "    # Predictions on the test set\n",
    "    y_pred = bagging_model.predict(X_test)\n",
    "\n",
    "    # Compute metrics for this fold\n",
    "    fold_mae = mean_absolute_error(y_test, y_pred)\n",
    "    fold_mse = mean_squared_error(y_test, y_pred)\n",
    "    fold_mape = mean_absolute_percentage_error(y_test, y_pred)\n",
    "\n",
    "    cv_mae.append(fold_mae)\n",
    "    cv_mse.append(fold_mse)\n",
    "    cv_mape.append(fold_mape)\n",
    "\n",
    "# Calculate mean of metrics across folds\n",
    "mean_cv_mae = np.mean(cv_mae)\n",
    "mean_cv_mse = np.mean(cv_mse)\n",
    "mean_cv_mape = np.mean(cv_mape)\n",
    "\n",
    "\n",
    "print('Mean Cross-validated MAE:', mean_cv_mae)\n",
    "print('Mean Cross-validated RMSE:', np.sqrt(mean_cv_mse))\n",
    "print('Mean Cross-validated MAPE:', mean_cv_mape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Cross-validated MAE: 348.05180352102195\n",
      "Mean Cross-validated RMSE: 1966.075669664016\n",
      "Mean Cross-validated MAPE: 0.026802741785782062\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import cross_val_score, KFold\n",
    "kf = KFold(n_splits=10, shuffle=True, random_state=69)\n",
    "cv_mae = []\n",
    "cv_mse = []\n",
    "cv_mape = []\n",
    "for train_index, test_index in kf.split(X_scaled):\n",
    "    X_train, X_test = X_scaled[train_index], X_scaled[test_index]\n",
    "    y_train, y_test = y.iloc[train_index], y.iloc[test_index]\n",
    "\n",
    "    # Fit the BaggingRegressor model\n",
    "    model.fit(X_train, y_train)\n",
    "\n",
    "    # Predictions on the test set\n",
    "    y_pred = model.predict(X_test)\n",
    "\n",
    "    # Compute metrics for this fold\n",
    "    fold_mae = mean_absolute_error(y_test, y_pred)\n",
    "    fold_mse = mean_squared_error(y_test, y_pred)\n",
    "    fold_mape = mean_absolute_percentage_error(y_test, y_pred)\n",
    "\n",
    "    cv_mae.append(fold_mae)\n",
    "    cv_mse.append(fold_mse)\n",
    "    cv_mape.append(fold_mape)\n",
    "\n",
    "# Calculate mean of metrics across folds\n",
    "mean_cv_mae = np.mean(cv_mae)\n",
    "mean_cv_mse = np.mean(cv_mse)\n",
    "mean_cv_mape = np.mean(cv_mape)\n",
    "\n",
    "\n",
    "print('Mean Cross-validated MAE:', mean_cv_mae)\n",
    "print('Mean Cross-validated RMSE:', np.sqrt(mean_cv_mse))\n",
    "print('Mean Cross-validated MAPE:', mean_cv_mape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\nikit\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\xgboost\\core.py:160: UserWarning: [13:34:13] WARNING: C:\\buildkite-agent\\builds\\buildkite-windows-cpu-autoscaling-group-i-0b3782d1791676daf-1\\xgboost\\xgboost-ci-windows\\src\\c_api\\c_api.cc:1240: Saving into deprecated binary model format, please consider using `json` or `ubj`. Model format will default to JSON in XGBoost 2.2 if not specified.\n",
      "  warnings.warn(smsg, UserWarning)\n"
     ]
    }
   ],
   "source": [
    "# Save the model to the disk\n",
    "model.save_model('model.bin')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}