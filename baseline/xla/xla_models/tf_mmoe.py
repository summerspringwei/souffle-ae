import sys, os
sys.path.extend(['', '/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/contrib/',
'/home/xiachunwei/Software/clean_tvm/tvm/python',
 '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/resnext',
 '/home/xiachunwei/Software/anaconda3/lib/python37.zip', '/home/xiachunwei/Software/anaconda3/lib/python3.7',
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload', '/home/xiachunwei/.local/lib/python3.7/site-packages',
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages',
 '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'])
sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python/")
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

@tf.function(jit_compile=True)
def tf_MMoE(inputs, expert_kernels, expert_bias, gate_kernels, gate_bias, units):
  # For evaluation
  # inputs = tf.constant(np_inputs, tf.float32)
  # expert_kernels = tf.constant(expert_kernels, tf.float32)
  # expert_bias = tf.constant(expert_bias, tf.float32)
  # gate_kernels = [tf.constant(gate_kernels[i], tf.float32)  for i in range(len(gate_kernels))]
  # gate_bias = [tf.constant(gate_bias[i], tf.float32) for i in range(len(gate_bias))]
  # inputs = tf.compat.v1.placeholder(tf.float32, (batch, input_dim))
  # expert_kernels = tf.compat.v1.placeholder(tf.float32, (input_dim, units, num_experts))
  # expert_bias = tf.compat.v1.placeholder(tf.float32, (units, num_experts))
  # gate_kernels = [tf.compat.v1.placeholder(tf.float32, (input_dim, num_experts)) for i in range(num_tasks)]
  # gate_bias = [tf.compat.v1.placeholder(tf.float32, (num_experts,)) for i in range(num_tasks)]
  gate_outputs = []
  final_outputs = []

  expert_outputs = tf.tensordot(inputs, expert_kernels, axes=1)
  expert_outputs = K.bias_add(expert_outputs, expert_bias)
  expert_outputs = tf.nn.relu(expert_outputs)
  for index, gate_kernel in enumerate(gate_kernels):
    gate_output = K.dot(x=inputs, y=gate_kernel)
    gate_output = K.bias_add(x=gate_output, bias=gate_bias[index])
    gate_output = K.softmax(gate_output, )
    gate_outputs.append(gate_output)

  for gate_output in gate_outputs:
    expanded_gate_output = K.expand_dims(gate_output, axis=1)
    weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, units, axis=1)
    final_outputs.append(K.sum(weighted_expert_output, axis=2))

  return final_outputs


def run_tf_mmoe(batch, input_dim, units, num_experts, num_tasks):
  import tensorflow as tf
  sys.path.insert(0, '/home/xiachunwei/Software/tensor-compiler/src/ocv')
  from tf_utils import tf_freeze_keras_model
  np_expert_kernels = np.random.randn(input_dim, units, num_experts).astype(np.float32)
  np_expert_bias = np.random.randn(units, num_experts).astype(np.float32)
  np_gate_kernels = [np.random.randn(input_dim, num_experts).astype(np.float32) for i in range(num_tasks)]
  np_gate_bias = [np.random.randn(num_experts).astype(np.float32) for i in range(num_tasks)]
  image = tf.zeros([1, input_dim])
  output = tf_MMoE(image, np_expert_kernels, np_expert_bias, np_gate_kernels, np_gate_bias, units)
  print(output)


def tf_freeze_MMoE(batch, input_dim, units, num_experts, num_tasks):
  import tensorflow as tf
  sys.path.insert(0, '/home/xiachunwei/Software/tensor-compiler/src/ocv')
  from tf_utils import tf_freeze_keras_model
  np_expert_kernels = np.random.randn(input_dim, units, num_experts).astype(np.float32)
  np_expert_bias = np.random.randn(units, num_experts).astype(np.float32)
  np_gate_kernels = [np.random.randn(input_dim, num_experts).astype(np.float32) for i in range(num_tasks)]
  np_gate_bias = [np.random.randn(num_experts).astype(np.float32) for i in range(num_tasks)]
  image = tf.keras.Input(shape=(input_dim, ), batch_size=1)
  output = tf_MMoE(image, np_expert_kernels, np_expert_bias, np_gate_kernels, np_gate_bias, units)
  model = tf.keras.Model(inputs=image, outputs=output)
  model_folder = os.path.join("/home/xiachunwei", "models", "tf_MMoE")
  model_name = "tf_MMoE_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  tf_freeze_keras_model(model, model_folder, model_name)
  sh_cmd = "python3 -m tf2onnx.convert --input {} --output {} --inputs {} --outputs {}".format(\
    os.path.join(model_folder, model_name+".pb"), os.path.join(model_folder, "{}.onnx".format(model_name)),\
      "x:0", "Identity:0")
  os.system(sh_cmd)
  sh_cmd = "trtexec --onnx={} --saveEngine={}".format(\
    os.path.join(model_folder, "{}.onnx".format(model_name)), os.path.join(model_folder, "{}_engine.trt".format(model_name)))
  os.system(sh_cmd)


if __name__=="__main__":
  run_tf_mmoe(1,10,16,8,2)
