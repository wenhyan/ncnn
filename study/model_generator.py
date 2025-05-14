import torch
import torch.nn as nn
import pnnx

class StudyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)
    


if __name__ == "__main__":
    model = StudyModel()
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     "study_relu.onnx",
    #     input_names=["input"],
    #     output_names=["output"],
    #     opset_version=12,
    #     do_constant_folding=True
    # )

    torch.save(model, "study_relu.pth   ")

    opt_model = pnnx.export(model, "study_relu.pth", dummy_input)
    result = opt_model(dummy_input) 

