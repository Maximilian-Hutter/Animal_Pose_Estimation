hparams = {

    # Base Settings
    "seed": 123,
    "gpus": 1,
    "gpu_mode": True,  #True
    "crop_size": None,
    "resume": False,
    "train_data_path": "C:/Data/ap-10k/",
    "augment_data": False,
    "threads": 0,
    "start_epoch": 0,
    "save_folder": "./weights/",
    "model_type": "Dehaze",
    "snapshots": 10,
    "pseudo_alpha": 1,
    "hazy_alpha": 1,

    #Train Settings
    "epochs": 200,
    "batch_size": 1,
    "gen_lambda": 1,
    "lr":2e-04,
    "beta1": 0.9595,
    "beta2": 0.9901,

    # Model Settings
    #"features": (8,16,32,64),
    "features": (4,4,8,16),
    "num_joints":17,
    "feature_out":1,
    "num_conv":1,
    "use_csdlkcb":True,
    "activation":"LeakyReLU",
    "kernelout":3
    #features=(8,16,32,64),num_joints=18,feature_out=1,num_conv=3,use_csdlkcb=True,activation="LeakyReLU",kernelout=3
    
}