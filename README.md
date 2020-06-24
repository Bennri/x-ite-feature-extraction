# Feature Extraction - X-ITE Pain Database [[1]](#ref_gruss)

## Feature Extraction Config Example
Create a file named `fe_config.json` in the root directory of this repository, containing the follwing content:<br>
`x_ite_ed_bio_path` - path to the bio signal data of the X-ITE Pain Database
`path_to_store_dataset` - path where to store extracted featureset
`dataset_file_name` - extension '.csv' will be added during the process<br>
`n_jobs` - number of cores to use during parallel processing<br>
`parallel_backend` - backend for python joblib module
```
{
  "x_ite_ed_bio_path": "/path/to/x-ite/dataset/Bio",
  "path_to_store_dataset": "/destination/to/extracted/dataset/",
  "dataset_file_name": "example_dataset_filename",
  "n_jobs": 32,
  "parallel_backend": "loky"
}
```


## References
<a name='ref_gruss'>[1] Gruss, S., Geiger, M., Werner, P., Wilhelm, O., Traue, H. C., Al-Hamadi, A., Walter, S. Multi-Modal Signals for Analyzing Pain Responses to Thermal and Electrical Stimuli. J. Vis. Exp. (146), e59057, doi:10.3791/59057 (2019).</a>
