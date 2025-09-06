
class Options():
    def __init__(self):
        super().__init__()
        self.Seed = 1234
        self.Epoch = 400
        self.Learning_Rate = 2e-4
        self.Batch_Size_Train = 6
        self.Batch_Size_Val = 6
        self.Patch_Size_Train = 128
        self.Patch_Size_Val = 128

        self.Input_Path_Train = '../Dataset/Snow100K/Train/Snow'
        self.Target_Path_Train = '../Dataset/Snow100K/Train/GT'

        self.Input_Path_Val = '../Dataset/Snow100K/Val/Snow'
        self.Target_Path_Val = '../Dataset/Snow100K/Val/GT'

        self.Dataset_Names = [
                              'FoggyCityscapes',
                              'RainCityscapes',
                              'RSCityscapes',
                              'SnowTrafficData',
                              'LowLightTrafficData',
                              'RainDS-syn'
        ]
        self.Path_Test = './AWTD/test'

        self.MODEL_RESUME_PATH = './model_best.pth'

        self.Num_Works = 4
        self.CUDA_USE = True