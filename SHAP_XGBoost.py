import shap

# Fit XGBoost again for SHAP
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum())
xgb.fit(X_train, y_train)

# SHAP analysis
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)
