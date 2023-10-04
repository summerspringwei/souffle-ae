
import sys
sys.path.extend(['', '/home/xiachunwei/Software/0.7-tvm/tvm/python', '/home/xiachunwei/Software/tensor-compiler/src/xla', '/home/xiachunwei/Software/anaconda3/lib/python37.zip', '/home/xiachunwei/Software/anaconda3/lib/python3.7', '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload', '/home/xiachunwei/Software/pytf2.10/lib/python3.7/site-packages', '/home/xiachunwei/.local/lib/python3.7/site-packages', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages', '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'])
import argparse
import tensorflow as tf
import numpy as np
import time


def get_name_and_shape(str_name_shape):
    name_shape_tuple_list = []
    com = str_name_shape.split(";")
    for c in com:
        if len(c) < 2:
            continue
        [name, str_shape] = c.split(":")
        s_list = [int(s) for s in str_shape.split(",")]
        name_shape_tuple_list.append((name, s_list))

    return name_shape_tuple_list


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


# @tf.function(jit_compile=True)
def load_and_run(model_file, inputs, outputs, dtype="float32"):
  tf.keras.backend.clear_session()
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  tf.config.optimizer.set_jit(True) # Enable XLA.
  with tf.device("/device:gpu:0"):
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    input_name_shape_list = get_name_and_shape(inputs)
    input_str_list = [name+":0" for (name, shape) in input_name_shape_list]

    # Wrap frozen graph to ConcreteFunctions
    outputs_str_list = [com+":0" for com in outputs.split(':')]

    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=input_str_list,
                                    outputs=outputs_str_list,
                                    print_graph=True)
    input_tensors  = []
    for name, shape in input_name_shape_list:
      tensor = tf.constant(np.random.random((shape)).astype(dtype), dtype=dtype, shape=shape, name=name)
      input_tensors.append(tensor)

    @tf.function(jit_compile=True)
    def _run_model():
      return frozen_func(*input_tensors)
    output = _run_model()
    print(output)
    return
    for i in range(100):
      output = _run_model()
      if i%100==0:
        print(i)
    print(output)
    start = time.time_ns()
    for i in range(1000):
      output = _run_model()
    print(output)
    end = time.time_ns()
    print("latency: ", (end-start)/1e6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument('--outputs', type=str, required=True)
    parser.add_argument("--dtype", type=str, )
    model_file = parser.parse_args().model_file
    model_outputs = parser.parse_args().outputs
    args = parser.parse_args()
    load_and_run(model_file, args.inputs, model_outputs, dtype=args.dtype)



if __name__ == '__main__':
    main()
