import onnx
from onnx import version_converter
import torch
import torch_mlir
from onnx2torch import convert

#onnx_model = onnx.load("swin-transformer/swin-transformer.onnx")
#onnx_model = onnx.load("mmoe/tf_MMoE_1_100_16_8_2.onnx")
#onnx_model = onnx.load("resnext/resnext_imagenet_101.onnx")
#onnx_model = onnx.load("bert/bert_1_384_768_3072.onnx")
#onnx_model = onnx.load("lstm/frozen_lstm_l8s8h256_bs1.onnx")
onnx_model = onnx.load("efficientnet/efficientnet-b0.onnx")
onnx_13_model = version_converter.convert_version(onnx_model, 11)
torch_model = convert(onnx_13_model)
#example_input = torch.zeros(1,3,224,224, dtype = torch.float32)
#example_input = torch.zeros(1,100, dtype = torch.float32)
#example_input = torch.zeros(1,3,224,224, dtype = torch.float32)
#example_input = torch.zeros(1,384,768, dtype = torch.float16)
#example_input = torch.zeros(8,1,256, dtype = torch.float32)
example_input = torch.zeros(1,3,224,224, dtype = torch.float32)
mlir = torch_mlir.compile(
    torch_model.cuda(0),
    example_input.cuda(0),
    output_type="linalg-on-tensors",
    use_tracing=True)
#with open("swin-transformer/swin-transformer.mlir", "w") as f:
#with open("mmoe/tf_MMoE_1_100_16_8_2.mlir", "w") as f:
#with open("resnext/resnext_imagenet_101.mlir", "w") as f:
#with open("bert/bert_1_384_768_3072.mlir", "w") as f:
#with open("lstm/frozen_lstm_l8s8h256_bs1.mlir", "w") as f:
with open("efficientnet/efficientnet-b0.mlir", "w") as f:
    f.write(str(mlir))
