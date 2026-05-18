import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/TrainingCode")
from initParam import inputParameters
from dataHandle import loadedFullDataset, returnDataset, returnTestDataset
from TrainLoop import trainLoop
import torch
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    params = inputParameters()
    
    print("Input parameters initialized successfully.")

    # trainDataset = loadedFullDataset(inputFileList= params.choosenFile, diffBand= params.diffBand, exceptBand= params.exceptBand, timeStamps= params.timeStamps, inputInfo= params.inputInfo, fullBand= params.fullBand)
    # valDataset = loadedFullDataset(inputFileList= params.choosenVal, diffBand= params.diffBand, exceptBand= params.exceptBand, timeStamps= params.timeStamps, inputInfo= params.inputInfo, fullBand= params.fullBand)
    fullDataset = loadedFullDataset(inputFileList= params.choosenFile, diffBand= params.diffBand, exceptBand= params.exceptBand, timeStamps= params.timeStamps, inputInfo= params.inputInfo, fullBand= params.fullBand)
    trainDataset, valDataset = train_test_split(fullDataset, test_size=0.2, random_state=42)
    print("Datasets loaded successfully.")

    train_dataset = returnDataset(
        trainDataset= trainDataset, exceptBand= params.exceptBand, device= params.device, batch_size= params.batch_size, timestamps= params.timeStamps, outputLabel= params.outputLabel, fullBand= params.fullBand
    )

    val_dataset = returnTestDataset(
        testDataFrame= valDataset, device= params.device, batch_size= params.batch_size, timestamps= params.timeStamps, outputLabel= params.outputLabel, exceptBand= params.exceptBand, fullBand= params.fullBand
    )

    print("Create train and validation dataloader successfully.")
    outputModel = trainLoop(
        epochs= params.epochs, my_model= params.myModel, device= params.device,
        loss_fn= params.loss_fn, optimizer= params.optimizer, scheduler= params.scheduler,
        outputLabel= params.outputLabel, fullBand= params.fullBand,
        train_dataset= train_dataset, val_dataset= val_dataset,
        wandbObject= params.run, threshold= params.threshold, metricInfo= params.outputMetrics, earlyStop= True, patience= params.patience,
        batch_size= params.batch_size
    ).trainBinaryOutput()

    
    # Save model
    if outputModel is not None:
        params.myModel.load_state_dict(outputModel.state_dict())
        torch.save(obj=params.myModel.state_dict(), f=params.model_dir)
        print("Save complete trained model")
    else:
        print("Model is none!")