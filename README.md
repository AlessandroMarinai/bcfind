The requirements are in the requirements.in file.

To run the DANN method you can call "python bcfind/train_DA.py bcfind/train_config_DA.yaml --gpu 0" specifying the correct gpu number on your machine.

To run the FDA method call "python bcfind/train_FDA.py bcfind/train_config_FDA.yaml --gpu 0" instead.

To run the DANN method on FDA-modified images run "python bcfind/train_DA_FDA.py bcfind/train_config_DA_FDA.yaml --gpu 0".

Add --only-test to skip the training part and only test the model. To choose the trained model specify the path of the experiment in the configuration or you can just keep the same .yaml file.
 
