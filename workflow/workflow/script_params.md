This file contains the parameters which need to be adjusted in the "## Settings ##" section for each script.

I_Dataset_Generation
- 1_generate_synthetic_streamflows.py
    - synthetic training dataset 
        - random_state = 42
        - n_runs = 1000
        - drought = None
    - synthetic drought datasets
        - random_state = 40
        - n_runs = 100
        - drought = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] (run script once for each different drought level)
- 2_simulate_water_shortages.py
    - synthetic training dataset 
        - drought = None
    - synthetic drought datasets
        - drought = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] (run script once for each different drought level)
- 3_process_datasets_for_ml.py
    - No adjustments necessary. Script runs once for all 6 datasets.

II_Machine_Learning
- 1_optimize_model.py
    - random_seed = 42
    - n_trials = 1000
- 2_train_model.py
    - random_seed = 8888
    - model config should be loaded from `repo_data/ml-data/configs/optimized_params.yaml`.
- 3_generate_predictions.py
    - run_name = "run_XXXXXXXX-YYYYYY" <--- Replace this with the name of the folder containing the trained model in II-2.
    - checkpoint_id = "249"

III_Results
- 1_plot_datasets.py
    - run_name = "run_XXXXXXXX-YYYYYY" <--- Replace this with the name of the folder containing the trained model in II-2.
    - checkpoint_id = "249"
- 2_plot_metadata.py
    - No adjustments necessary
- 3_plot_flowchart_components.py
    - No adjustments necessary
- 4_plot_results.py
    - run_name = "run_XXXXXXXX-YYYYYY" <--- Replace this with the name of the folder containing the trained model in II-2.
    - checkpoint_id = "249"
