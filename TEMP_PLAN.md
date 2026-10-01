# Replace DFT comparison MSE with RMSE

Reuse paired error sums; export sqrt(mean squared error) with unsquared units.
Change official metrics, standalone comparison export, tests and documentation.
Keep internal squared sums for exact aggregate RMSE, remove public MSE fields.
Validate synthetic energy/force RMSE and missing-data behavior. No production data changes.
