
class Options():
    def __init__(self):
        super().__init__()
        self.Seed = 12345
        self.Epoch = 400
        self.Learning_Rate = 2e-4
        self.Batch_Size_Train = 6
        self.Batch_Size_Val = 6
        self.Patch_Size_Train = 128
        self.Patch_Size_Val = 128

        self.Input_Path_Train = '../Dataset/All-In-One(train-only)/Lq'
        self.Target_Path_Train = '../Dataset/All-In-One(train-only)/Gt'

        self.Input_Path_Val = '../Dataset/RSCityscapes/Lq'
        self.Target_Path_Val = '../Dataset/RSCityscapes/Gt'

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