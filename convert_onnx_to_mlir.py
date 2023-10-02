import torch
import torch_mlir
import io
import iree.compiler as ireec

model = # ... the model we're compiling
example_input = # ... an input to the model with the expected shape and dtype
mlir = torch_mlir.compile(
    model,
    example_input,
    output_type="linalg-on-tensors",
    use_tracing=True)

iree_backend = "llvm-cpu"
iree_input_type = "tm_tensor"
bytecode_stream = io.BytesIO()
mlir.operation.write_bytecode(bytecode_stream)
iree_vmfb = ireec.compile_str(bytecode_stream.getvalue(),
                              target_backends=[iree_backend],
                              input_type=iree_input_type)

