import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/TestCode")
from initParam import inputParameters
from TestLoop_Optimized import testLoopOptimized

if __name__ == "__main__":
    params = inputParameters()
    
    print("Input parameters initialized successfully.") 

    outputModel = testLoopOptimized(
        my_model= params.myModel,
        device= params.device,
        loss_fn= params.loss_fn,
        outputLabel= params.outputLabel,
        diffBand= params.diffBand,
        fullBand= params.fullBand,
        exceptBand= params.exceptBand,
        listTestFile= params.choosenFile,
        wandbObject= params.run,
        threshold= params.threshold,
        metricInfo= params.metricInfo,
        batch_size= params.batch_size,
        timeStamps= params.timeStamps,
        inputInfo= params.inputInfo,
        prefetch_size=3
    ).testBinaryOutput(params.metricInfo)