import os
import torch
import wandb
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/TestCode")
from Model import LSTM
from FocalLoss import FocalLoss

class inputParameters:
    def __init__(self):
        # Test data folder
        self.inputFolder = "/sdd/Dubaoset/src/Phong/Model/data/testClean"
        # self.inputInfo = "/sdd/Dubaoset/src/Phong/Model/TrainingCode/infoJsonl/total_456_scales.jsonl"

        # self.choosenFile = [
        #     os.path.join(self.inputFolder, item) for item in os.listdir(self.inputFolder) if item.endswith('.parquet')
        # ]
        self.inputInfo = "/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/json_scale/total_scales.jsonl"

        self.choosenFile = [
            "/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/test_data.parquet"
        ]

        # self.fullBand = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI', 'Dem_value', 'NDVIIsLand', 'DEMIsLand']
        self.fullBand = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI', 'Dem_value']
        self.diffBand = None
        self.exceptBand = ['Dem_value', 'DEMIsLand']

        self.outputLabel = ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"]
        self.timeStamps = 6

        self.metricInfo = "/sdd/Dubaoset/src/Thang/TrainProcessMB/log_test_process.log"
        self.modelName = "trainDataMB.pth"

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        
        self.lr = 0.0001
        self.weight_decay = 0.0001
        self.batch_size = 512
        self.threshold = 0.5
        self.hidden_size= 256
        self.dropout= 0.2
        self.num_layer= 2
        self.output_features= 1
        self.alpha = 0.8
        self.gamma = 2

        wandb.login(key="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")
        self.run = wandb.init(
            project="testNewDistribution",
            # Track hyperparameters and run metadata.
            config={
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "hidden_size": self.hidden_size,
                "num_layer": self.num_layer,
                "input_size": len(self.fullBand),
                "output_features": self.output_features,
            }
        )
        torch.manual_seed(42)
        self.myModel = LSTM(
            input_size=len(self.fullBand),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layer=self.num_layer,
            output_features=self.output_features
        ).to(self.device)

        self.model_dir = os.path.join("/sdd/Dubaoset/src/Thang/TrainProcessMB/model", self.modelName)
        self.myModel.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        self.loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        
